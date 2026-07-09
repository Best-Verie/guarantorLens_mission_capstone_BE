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

# Shared loan-risk index: score every loan once (model + flag overlay), remember each loan's
# band/score/exposure and which loans each member backs. Powers both the contagion view and the
# weak-links ranking, so they never disagree and we only score the book once.
_RISK_CACHE = {"at": 0.0, "per_loan": None, "by_guar": None}


def _risk_index():
    now = time.time()
    if _RISK_CACHE["per_loan"] is not None and now - _RISK_CACHE["at"] < _EW_TTL:
        return _RISK_CACHE["per_loan"], _RISK_CACHE["by_guar"]
    loans = network_data.LOANS
    inputs = [{
        "amount": ln.get("amount", 0) or 0,
        "savings": scoring.MEMBERS.get(ln.get("borrower"), {}).get("savings"),
        "salary": scoring.MEMBERS.get(ln.get("borrower"), {}).get("salary"),
        "disb_date": ln.get("disb_date"), "guarantor_ids": ln.get("guarantors", []),
        "borrower_id": ln.get("borrower"),
    } for ln in loans]
    scores = scoring.score_many(inputs)
    per_loan, by_guar = {}, {}
    for ln, (prob, band) in zip(loans, scores):
        gs = ln.get("guarantors", []) or []
        rec = {
            "loan_key": ln["loan_key"], "borrower": ln.get("borrower"),
            "borrower_uid": scoring.member_uid(ln.get("borrower")),
            "amount": ln.get("amount", 0) or 0,
            "band": scoring.adjust_band(band, gs, ln.get("borrower")),
            "score": scoring._display_score(prob),
        }
        per_loan[ln["loan_key"]] = rec
        for g in gs:
            by_guar.setdefault(g, []).append(ln["loan_key"])
    _RISK_CACHE.update(at=now, per_loan=per_loan, by_guar=by_guar)
    return per_loan, by_guar


@router.get("/insights/weak-links")
def weak_links(limit: int = 12, user: User = Depends(get_current_user)):
    """Members who are 'single points of failure' in the guarantee network: they back many
    loans, so if they fail, a lot wobbles at once. Ranked by number of loans backed, with the
    total exposure and how many of those loans are already high risk."""
    per_loan, by_guar = _risk_index()
    rows = []
    for mid, keys in by_guar.items():
        if len(keys) < 5:            # only the genuinely concentrated backers
            continue
        loans = [per_loan[k] for k in keys]
        exposure = sum(x["amount"] for x in loans)
        high = sum(1 for x in loans if x["band"] == "High")
        m = scoring.MEMBERS.get(mid, {})
        rows.append({
            "member_id": mid, "uid": scoring.member_uid(mid),
            "branch": m.get("branch"), "ever_defaulted": bool(m.get("ever_defaulted", 0)),
            "loans_backed": len(keys), "high_risk": high, "exposure": exposure,
        })
    rows.sort(key=lambda r: (r["loans_backed"], r["exposure"]), reverse=True)
    return rows[:max(1, min(limit, 50))]


@router.get("/watchlist", response_model=List[WatchlistItem])
def watchlist(user: User = Depends(get_current_user)):
    items = network_data.watchlist(scoring.MEMBERS)
    for it in items:
        it["borrower_uid"] = scoring.member_uid(it.get("borrower"))
    return items


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
    rows = network_data.super_guarantors(scoring.MEMBERS)
    for r in rows:
        r["uid"] = scoring.member_uid(r.get("member_id"))
    return rows


@router.get("/member/{ref}/contagion")
def contagion(ref: str, user: User = Depends(get_current_user)):
    """If this member (as a guarantor) defaulted, what is exposed? Lists the loans they back,
    each loan's current risk, and the totals - the 'ripple' behind the network story."""
    member_id = scoring.resolve_member_ref(ref)
    if member_id is None:
        return {"member_id": ref, "loans_backed": 0, "high_risk": 0, "exposure": 0.0, "loans": []}
    per_loan, by_guar = _risk_index()
    keys = by_guar.get(member_id, [])
    loans = sorted((per_loan[k] for k in keys), key=lambda x: x["score"], reverse=True)
    return {
        "member_id": member_id, "uid": scoring.member_uid(member_id),
        "loans_backed": len(loans),
        "high_risk": sum(1 for x in loans if x["band"] == "High"),
        "medium_risk": sum(1 for x in loans if x["band"] == "Medium"),
        "exposure": sum(x["amount"] for x in loans),
        "loans": loans[:12],
    }


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
        "loan_key": ln["loan_key"], "borrower": ln.get("borrower"),
        "borrower_uid": scoring.member_uid(ln.get("borrower")), "branch": ln.get("branch"),
        "amount": ln.get("amount", 0), "days_in_arrears": int(ln.get("days_in_arrears", 0) or 0),
        "risk_score": round(prob * 100),
        # same leak-free flag overlay as the assessment card
        "band": scoring.adjust_band(band, ln.get("guarantors", []), ln.get("borrower")),
    } for ln, (prob, band) in zip(active, scores)]
    items.sort(key=lambda x: x["risk_score"], reverse=True)
    _EW_CACHE["data"] = items[:300]
    _EW_CACHE["at"] = now
    return _EW_CACHE["data"]
