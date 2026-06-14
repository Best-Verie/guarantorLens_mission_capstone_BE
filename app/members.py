"""Member lookup endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status

from . import scoring
from .auth import get_current_user
from .models import User
from .schema import MemberProfile

router = APIRouter(tags=["members"])


@router.get("/member/{member_id}", response_model=MemberProfile)
def get_member(member_id: str, user: User = Depends(get_current_user)):
    m = scoring.MEMBERS.get(member_id)
    if not m:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
    return MemberProfile(
        member_id=m["member_id"],
        branch=m.get("branch"),
        savings=m.get("savings"),
        salary=m.get("salary"),
        ever_defaulted=bool(m.get("ever_defaulted", 0)),
        default_date=m.get("default_date"),
        loans_backed=int(m.get("loans_backed", 0)),
        total_connections=int(m.get("degree", 0)),
        community_default_rate=float(m.get("community_default_rate", 0.0)),
    )
