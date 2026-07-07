"""Portfolio insights: watchlist, super-guarantors, default communities."""
import time
from typing import List

from fastapi import APIRouter, Depends

from . import network_data, scoring
from .auth import get_current_user
from .models import User
from .schema import (
    CommunityStat, EarlyWarningItem, InsightsOverview, SuperGuarantor, WatchlistItem,
)

router = APIRouter(tags=["insights"])

# Portfolio numbers are computed from the static loan/member tables, so cache them.
_OV_CACHE = {"data": None}

# Early warning scores every active loan, which is the slow part on a small host. The
# inputs are static per process, so cache the ranked result for a while. Same for every
# user, so a single global entry is fine.
_EW_CACHE = {"at": 0.0, "data": None}
_EW_TTL = 600  # seconds


@router.get("/watchlist", response_model=List[WatchlistItem])
def watchlist(user: User = Depends(get_current_user)):
    return network_data.watchlist(scoring.MEMBERS)


@router.get("/insights/overview", response_model=InsightsOverview)
def overview(user: User = Depends(get_current_user)):
    """Key portfolio numbers you can read off the loan/member tables: size, outcomes,
    money at risk, and the shape of the guarantor network."""
    if _OV_CACHE["data"] is not None:
        return _OV_CACHE["data"]

    loans = network_data.LOANS
    members = list(scoring.MEMBERS.values())
    M = scoring.MEMBERS
    bad_outcomes = {"Written off", "In arrears 90+"}

    outcomes, branches = {}, {}
    total_disbursed = written_off_value = arrears_value = 0.0
    n_arrears = matured = n_bad = 0
    gcounts, uniq_g = [], set()
    loans_backed_by_defaulter = 0
    for ln in loans:
        amt = float(ln.get("amount") or 0)
        total_disbursed += amt
        oc = ln.get("outcome") or "Active"
        outcomes[oc] = outcomes.get(oc, 0) + 1
        br = ln.get("branch") or "Unknown"
        branches[br] = branches.get(br, 0) + 1
        if oc != "Active":
            matured += 1
            if oc in bad_outcomes:
                n_bad += 1
        if oc == "Written off":
            written_off_value += amt
        if int(ln.get("days_in_arrears") or 0) > 0:
            n_arrears += 1
            arrears_value += amt
        guars = ln.get("guarantors", []) or []
        gcounts.append(len(guars))
        uniq_g.update(guars)
        if any(M.get(g, {}).get("ever_defaulted") == 1 for g in guars):
            loans_backed_by_defaulter += 1

    over = sum(1 for m in members if (m.get("loans_backed") or 0) >= scoring.FLAG_TH["over_committed_loads"])
    defaulters = sum(1 for m in members if m.get("ever_defaulted") == 1)
    comms = {m.get("community_id") for m in members if m.get("community_id")}
    worst = max((m.get("community_default_rate") or 0.0) for m in members) if members else 0.0
    n_loans = len(loans)

    data = InsightsOverview(
        n_loans=n_loans, n_members=len(members), total_disbursed=total_disbursed,
        outcomes=outcomes, branches=branches,
        bad_rate=(n_bad / matured) if matured else 0.0,
        written_off_value=written_off_value,
        n_arrears=n_arrears, arrears_value=arrears_value,
        unique_guarantors=len(uniq_g),
        avg_guarantors=(sum(gcounts) / len(gcounts)) if gcounts else 0.0,
        over_committed=over, ever_defaulted=defaulters,
        loans_backed_by_defaulter=loans_backed_by_defaulter,
        pct_backed_by_defaulter=(loans_backed_by_defaulter / n_loans) if n_loans else 0.0,
        n_communities=len(comms), worst_community_rate=worst,
    )
    _OV_CACHE["data"] = data
    return data


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
    now = time.time()
    if _EW_CACHE["data"] is not None and now - _EW_CACHE["at"] < _EW_TTL:
        return _EW_CACHE["data"]
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
    _EW_CACHE["data"] = items[:300]
    _EW_CACHE["at"] = now
    return _EW_CACHE["data"]
