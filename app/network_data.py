"""Loads the loans/edges table and builds member-centric indexes for the
member and network views. Falls back to empty indexes if the file is absent."""
import json
import os
from collections import defaultdict

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
LOANS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_loans.json")

# Cap how many neighbours we draw, so a heavy backer's graph stays readable.
MAX_NEIGHBOURS = 40


def _load():
    try:
        with open(LOANS_PATH) as fh:
            loans = json.load(fh)
    except Exception:
        loans = []
    by_borrower = defaultdict(list)   # member -> loans they took
    by_guarantor = defaultdict(list)  # member -> loans they guarantee
    for ln in loans:
        by_borrower[ln["borrower"]].append(ln)
        for g in ln.get("guarantors", []):
            by_guarantor[g].append(ln)
    return loans, by_borrower, by_guarantor


LOANS, BY_BORROWER, BY_GUARANTOR = _load()

HAS_LOANS = bool(LOANS)


def reload():
    """Re-read the loans table after the admin uploads new artifacts."""
    global LOANS, BY_BORROWER, BY_GUARANTOR, HAS_LOANS
    LOANS, BY_BORROWER, BY_GUARANTOR = _load()
    HAS_LOANS = bool(LOANS)
    return len(LOANS)


def _outcome(ln: dict) -> str:
    """Display outcome, tolerant of both schemas (old had 'outcome'; new has label/troubled)."""
    if ln.get("outcome"):
        return ln["outcome"]
    if ln.get("label") == 1:
        return "Written off"
    if int(ln.get("days_in_arrears") or 0) >= 90 or ln.get("troubled") == 1:
        return "In arrears"
    return "Active"


def member_detail(member_id: str, members: dict) -> dict:
    """Return loans, backers, guarantees-given, and an ego network for a member."""
    own_loans = BY_BORROWER.get(member_id, [])
    backed = BY_GUARANTOR.get(member_id, [])

    loans = [
        {
            "loan_key": ln.get("loan_key"),
            "amount": ln.get("amount", 0),
            "disb_date": ln.get("disb_date"),
            "outcome": _outcome(ln),
            "guarantors": ln.get("guarantors", []),
        }
        for ln in own_loans
    ]

    # Unique members who back this member's loans.
    backers = []
    seen = set()
    for ln in own_loans:
        for g in ln.get("guarantors", []):
            if g not in seen:
                seen.add(g)
                backers.append(g)

    guarantees_given = [
        {"loan_key": ln.get("loan_key"), "borrower": ln.get("borrower"), "outcome": _outcome(ln)}
        for ln in backed
    ]

    # Ego network: backers point to the member, the member points to those they back.
    def node(mid, role):
        m = members.get(mid, {})
        return {
            "id": mid,
            "role": role,
            "ever_defaulted": bool(m.get("ever_defaulted", 0)),
            "loans_backed": int(m.get("loans_backed", 0)),
        }

    nodes = {member_id: node(member_id, "self")}
    edges = []
    for g in backers[:MAX_NEIGHBOURS]:
        nodes.setdefault(g, node(g, "backer"))
        edges.append({"source": g, "target": member_id})
    backed_borrowers = []
    for ln in backed:
        b = ln["borrower"]
        if b not in backed_borrowers:
            backed_borrowers.append(b)
    for b in backed_borrowers[:MAX_NEIGHBOURS]:
        nodes.setdefault(b, node(b, "backed"))
        edges.append({"source": member_id, "target": b})

    return {
        "loans": loans,
        "backers": backers,
        "guarantees_given": guarantees_given,
        "network": {"nodes": list(nodes.values()), "edges": edges},
    }


MAX_NEIGHBOURHOOD = 50


def neighborhood(member_id: str, members: dict) -> dict:
    """A member's 1-hop neighborhood with every guarantee edge among those members,
    so the graph shows real interconnections, not just a star."""
    own_loans = BY_BORROWER.get(member_id, [])
    backed = BY_GUARANTOR.get(member_id, [])

    backers = []
    for ln in own_loans:
        backers.extend(ln.get("guarantors", []))
    backed_borrowers = [ln["borrower"] for ln in backed]

    nbr = [member_id]
    for m in backers + backed_borrowers:
        if m not in nbr:
            nbr.append(m)
    nbr = nbr[:MAX_NEIGHBOURHOOD]
    nbr_set = set(nbr)

    def role(mid):
        if mid == member_id:
            return "self"
        if mid in backers:
            return "backer"
        return "backed"

    def node(mid):
        m = members.get(mid, {})
        return {
            "id": mid,
            "role": role(mid),
            "ever_defaulted": bool(m.get("ever_defaulted", 0)),
            "loans_backed": int(m.get("loans_backed", 0)),
        }

    # All guarantee edges where both endpoints are in the neighborhood.
    cand = {}
    for m in nbr_set:
        for ln in BY_BORROWER.get(m, []):
            cand[ln["loan_key"]] = ln
        for ln in BY_GUARANTOR.get(m, []):
            cand[ln["loan_key"]] = ln
    edges, seen = [], set()
    for ln in cand.values():
        b = ln["borrower"]
        if b not in nbr_set:
            continue
        for g in ln.get("guarantors", []):
            if g in nbr_set and (g, b) not in seen:
                seen.add((g, b))
                edges.append({"source": g, "target": b})

    return {"center": member_id, "nodes": [node(m) for m in nbr], "edges": edges}


# --- portfolio insights -----------------------------------------------------

def watchlist(members: dict, min_days: int = 90, limit: int = 300) -> list:
    """Active loans that are min_days+ in arrears, most at risk first."""
    items = []
    for ln in LOANS:
        if ln.get("payment_status", "").lower() == "active" and ln.get("days_in_arrears", 0) >= min_days:
            gids = ln.get("guarantors", [])
            items.append({
                "loan_key": ln["loan_key"],
                "borrower": ln["borrower"],
                "branch": ln.get("branch"),
                "amount": ln["amount"],
                "days_in_arrears": ln.get("days_in_arrears", 0),
                "backed_by_defaulter": any(members.get(g, {}).get("ever_defaulted") == 1 for g in gids),
            })
    items.sort(key=lambda x: x["days_in_arrears"], reverse=True)
    return items[:limit]


def super_guarantors(members: dict, limit: int = 20) -> list:
    """Members backing the most loans (most systemic exposure if they fail)."""
    bad_backed = defaultdict(int)
    for ln in LOANS:
        if ln.get("label") == 1:
            for g in ln.get("guarantors", []):
                bad_backed[g] += 1
    rows = []
    for mid, m in members.items():
        lb = int(m.get("loans_backed", 0))
        if lb <= 0:
            continue
        rows.append({
            "member_id": mid,
            "branch": m.get("branch"),
            "loans_backed": lb,
            "ever_defaulted": bool(m.get("ever_defaulted", 0)),
            "bad_loans_backed": int(bad_backed.get(mid, 0)),
        })
    rows.sort(key=lambda x: x["loans_backed"], reverse=True)
    return rows[:limit]


def communities(members: dict, min_size: int = 5, limit: int = 30) -> list:
    """Guarantee communities ranked by their historical default rate."""
    groups = defaultdict(list)
    for m in members.values():
        cid = m.get("community_id")
        if cid:
            groups[cid].append(m)
    out = []
    for cid, ms in groups.items():
        size = len(ms)
        if size < min_size:
            continue
        dr = sum(1 for x in ms if x.get("ever_defaulted") == 1) / size
        out.append({"community_id": cid, "branch": ms[0].get("branch"), "size": size, "default_rate": round(dr, 4)})
    out.sort(key=lambda x: x["default_rate"], reverse=True)
    return out[:limit]
