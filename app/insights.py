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
    active = [ln for ln in network_data.LOANS
              if str(ln.get("payment_status", "")).lower() == "active"
              and int(ln.get("days_in_arrears", 0) or 0) < 90][:1500]   # bound the work
    inputs = [{
        "amount": ln.get("amount", 0),
        "savings": scoring.MEMBERS.get(ln.get("borrower"), {}).get("savings"),
        "salary": scoring.MEMBERS.get(ln.get("borrower"), {}).get("salary"),
        "disb_date": ln.get("disb_date"), "guarantor_ids": ln.get("guarantors", []),
        "borrower_id": ln.get("borrower"),
    } for ln in active]
    scores = scoring.score_many(inputs)   # one predict over all active loans
    items = [{
        "loan_key": ln["loan_key"], "borrower": ln.get("borrower"), "branch": ln.get("branch"),
        "amount": ln.get("amount", 0), "days_in_arrears": int(ln.get("days_in_arrears", 0) or 0),
        "risk_score": round(prob * 100),
        # same leak-free flag overlay as the assessment card
        "band": scoring.adjust_band(band, ln.get("guarantors", []), ln.get("borrower")),
    } for ln, (prob, band) in zip(active, scores)]
    items.sort(key=lambda x: x["risk_score"], reverse=True)
    return items[:300]
