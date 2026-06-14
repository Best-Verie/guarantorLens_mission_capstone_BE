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


def member_detail(member_id: str, members: dict) -> dict:
    """Return loans, backers, guarantees-given, and an ego network for a member."""
    own_loans = BY_BORROWER.get(member_id, [])
    backed = BY_GUARANTOR.get(member_id, [])

    loans = [
        {
            "loan_key": ln["loan_key"],
            "amount": ln["amount"],
            "disb_date": ln.get("disb_date"),
            "outcome": ln["outcome"],
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
        {"loan_key": ln["loan_key"], "borrower": ln["borrower"], "outcome": ln["outcome"]}
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
