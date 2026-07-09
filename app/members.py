"""Member lookup endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status

from . import network_data, scoring
from .auth import get_current_user
from .models import User
from .schema import MemberDetail, NetworkView

router = APIRouter(tags=["members"])


_SAMPLE_CACHE = {"data": None}


@router.get("/members/examples")
def member_examples(user: User = Depends(get_current_user)):
    """A few real member IDs and one ready-made, *presentable* assessment (a genuinely
    high-risk real loan where the model score and the band agree), pulled from whatever
    data is loaded so the UI examples always match the deployed dataset."""
    ids = list(scoring.MEMBERS.keys())[:6]
    if _SAMPLE_CACHE["data"] is None:
        best = None
        checked = 0
        for ln in network_data.LOANS:
            gs = ln.get("guarantors") or []
            b = ln.get("borrower")
            if b not in scoring.MEMBERS or not (2 <= len(gs) <= 3):
                continue
            m = scoring.MEMBERS[b]
            try:
                r = scoring.assess(ln.get("amount") or 0, m.get("savings"), m.get("salary"),
                                   ln.get("disb_date"), gs[:3], borrower_id=b, interest_rate=14)
            except Exception:
                continue
            checked += 1
            cand = {"borrower_id": b, "guarantor_ids": gs[:3], "amount": ln.get("amount") or 0,
                    "savings": m.get("savings"), "salary": m.get("salary"), "interest_rate": 14}
            if best is None or r["risk_score"] > best[0]:
                best = (r["risk_score"], cand)
            if r["risk_score"] >= 60 or checked >= 400:   # a clear High, model + band agree
                break
        _SAMPLE_CACHE["data"] = best[1] if best else None
    return {"member_ids": ids, "sample": _SAMPLE_CACHE["data"]}


@router.get("/members")
def list_members(q: str = "", branch: str = "", sort: str = "loans_backed",
                 order: str = "desc", page: int = 1, page_size: int = 25,
                 user: User = Depends(get_current_user)):
    """Browse/search the member directory. Server-side filter + sort + paginate so the UI
    can show a table over all ~10k members without shipping them all to the browser."""
    needle = (q or "").strip().lower()
    rows = list(scoring.MEMBERS.values())
    if needle:
        rows = [m for m in rows if needle in str(m.get("member_id", "")).lower()]
    if branch:
        rows = [m for m in rows if (m.get("branch") or "") == branch]

    keyf = {
        "loans_backed": lambda m: m.get("loans_backed") or 0,
        "savings": lambda m: m.get("savings") or 0,
        "salary": lambda m: m.get("salary") or 0,
        "member_id": lambda m: str(m.get("member_id", "")),
    }.get(sort, lambda m: m.get("loans_backed") or 0)
    rows.sort(key=keyf, reverse=(order != "asc"))

    total = len(rows)
    page = max(1, int(page))
    page_size = min(max(1, int(page_size)), 100)
    start = (page - 1) * page_size
    items = [{
        "member_id": m["member_id"],
        "uid": scoring.member_uid(m["member_id"]),
        "branch": m.get("branch"),
        "savings": m.get("savings"),
        "salary": m.get("salary"),
        "loans_backed": int(m.get("loans_backed") or 0),
        "ever_defaulted": bool(m.get("ever_defaulted", 0)),
    } for m in rows[start:start + page_size]]
    branches = sorted({m.get("branch") for m in scoring.MEMBERS.values() if m.get("branch")})
    return {"items": items, "total": total, "page": page, "page_size": page_size, "branches": branches}


@router.get("/network/{ref}", response_model=NetworkView)
def get_network(ref: str, user: User = Depends(get_current_user)):
    member_id = scoring.resolve_member_ref(ref)
    if member_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
    return network_data.neighborhood(member_id, scoring.MEMBERS)


@router.get("/member/{ref}", response_model=MemberDetail)
def get_member(ref: str, user: User = Depends(get_current_user)):
    # ref may be an opaque uid (preferred) or a raw account number (typed lookup / old link).
    member_id = scoring.resolve_member_ref(ref)
    m = scoring.MEMBERS.get(member_id) if member_id else None
    if not m:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
    detail = network_data.member_detail(member_id, scoring.MEMBERS)

    # opaque url ids for every member linked on this page (self, guarantors, backers, borrowers, nodes)
    refs = {member_id}
    refs.update(detail.get("backers", []))
    refs.update(g["borrower"] for g in detail.get("guarantees_given", []))
    for ln in detail.get("loans", []):
        refs.update(ln.get("guarantors", []))
    for n in detail.get("network", {}).get("nodes", []):
        refs.add(n["id"])
    uids = {r: scoring.member_uid(r) for r in refs if r}

    return MemberDetail(
        member_id=m["member_id"],
        uid=scoring.member_uid(member_id),
        uids=uids,
        branch=m.get("branch"),
        savings=m.get("savings"),
        salary=m.get("salary"),
        ever_defaulted=bool(m.get("ever_defaulted", 0)),
        default_date=m.get("default_date"),
        loans_backed=int(m.get("loans_backed", 0)),
        total_connections=int(m.get("degree", 0)),
        community_default_rate=float(m.get("community_default_rate", 0.0)),
        loans=detail["loans"],
        backers=detail["backers"],
        guarantees_given=detail["guarantees_given"],
        network=detail["network"],
    )
