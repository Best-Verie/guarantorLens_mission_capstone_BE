"""Member lookup endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status

from . import network_data, scoring
from .auth import get_current_user
from .models import User
from .schema import MemberDetail, NetworkView

router = APIRouter(tags=["members"])


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
