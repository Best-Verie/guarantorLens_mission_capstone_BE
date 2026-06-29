"""Portfolio insights: watchlist, super-guarantors, default communities."""
from typing import List

from fastapi import APIRouter, Depends

from . import network_data, scoring
from .auth import get_current_user
from .models import User
from .schema import CommunityStat, EarlyWarningItem, SuperGuarantor, WatchlistItem

router = APIRouter(tags=["insights"])


@router.get("/watchlist", response_model=List[WatchlistItem])
def watchlist(user: User = Depends(get_current_user)):
    return network_data.watchlist(scoring.MEMBERS)


@router.get("/insights/super-guarantors", response_model=List[SuperGuarantor])
def super_guarantors(user: User = Depends(get_current_user)):
    return network_data.super_guarantors(scoring.MEMBERS)


@router.get("/insights/communities", response_model=List[CommunityStat])
def communities(user: User = Depends(get_current_user)):
    return network_data.communities(scoring.MEMBERS)


@router.get("/insights/early-warning", response_model=List[EarlyWarningItem])
def early_warning(user: User = Depends(get_current_user)):
    """Active loans that are NOT yet 90 days late, scored by the model and ranked by
    predicted risk. This is the predictive monitoring (catch trouble before it shows),
    distinct from the watchlist (loans already in arrears)."""
    items = []
    for ln in network_data.LOANS:
        if str(ln.get("payment_status", "")).lower() != "active":
            continue
        if int(ln.get("days_in_arrears", 0) or 0) >= 90:
            continue
        borrower = ln.get("borrower")
        m = scoring.MEMBERS.get(borrower, {})
        prob, band = scoring.score_loan(
            ln.get("amount", 0), m.get("savings"), m.get("salary"),
            ln.get("disb_date"), ln.get("guarantors", []), borrower,
        )
        items.append({
            "loan_key": ln["loan_key"], "borrower": borrower, "branch": ln.get("branch"),
            "amount": ln.get("amount", 0), "days_in_arrears": int(ln.get("days_in_arrears", 0) or 0),
            "risk_score": round(prob * 100), "band": band,
        })
    items.sort(key=lambda x: x["risk_score"], reverse=True)
    return items[:300]
