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
