"""Loads the trained honest model and the member/network table, and scores a loan.

The model is the leakage-controlled ("honest") model exported from notebook 04.
Its features can all be rebuilt for a single new loan from the member table, so
no graph retraining is needed at serve time. If the model cannot be loaded
(missing file or library), the endpoint falls back to a transparent heuristic so
the API still responds.
"""
import json
import os
from bisect import bisect_left
from datetime import date

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_serving.joblib")
MEMBERS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_members.json")

_DEFAULT_BANDS = {"medium": 0.30, "high": 0.60}
_DEFAULT_FLAGS = {"over_committed_loads": 8, "high_default_community": 0.12}


def _load():
    members = {}
    try:
        with open(MEMBERS_PATH) as fh:
            members = {m["member_id"]: m for m in json.load(fh)}
    except Exception:
        members = {}
    try:
        import joblib
        bundle = joblib.load(MODEL_PATH)
        return (bundle["model"], bundle["features"], members,
                bundle.get("bands", _DEFAULT_BANDS),
                bundle.get("flag_thresholds", _DEFAULT_FLAGS), "model")
    except Exception:
        return (None, None, members, _DEFAULT_BANDS, _DEFAULT_FLAGS, "heuristic")


MODEL, FEATURES, MEMBERS, BANDS, FLAG_TH, _LOAD_SOURCE = _load()


def _member(mid):
    return MEMBERS.get(mid, {})


def _guarantee_dates(mid):
    return sorted(date.fromisoformat(x) for x in _member(mid).get("guarantee_dates", []))


def _band(p: float) -> str:
    if p >= BANDS["high"]:
        return "High"
    if p >= BANDS["medium"]:
        return "Medium"
    return "Low"


def _build_features(amount, savings, salary, disb, guarantor_ids):
    """Rebuild the honest model's features for one loan. Matches notebook 04."""
    import numpy as np

    loads = [bisect_left(_guarantee_dates(g), disb) for g in guarantor_ids] or [0]

    def prior(g):
        dt = _member(g).get("default_date")
        return 1.0 if dt and date.fromisoformat(dt) < disb else 0.0

    sav = [_member(g)["savings"] for g in guarantor_ids if _member(g).get("savings") is not None]
    sal = [_member(g)["salary"] for g in guarantor_ids if _member(g).get("salary") is not None]
    savings = savings or 0.0
    salary = salary or 0.0
    return {
        "log_amount": float(np.log1p(amount)),
        "savings": float(savings),
        "salary": float(salary),
        "loan_to_savings": amount / (savings + 1),
        "loan_to_salary": amount / (salary + 1),
        "n_guarantors": len(guarantor_ids),
        "g_prior_default_rate": float(np.mean([prior(g) for g in guarantor_ids])) if guarantor_ids else 0.0,
        "g_load_asof_mean": float(np.mean(loads)),
        "g_load_asof_max": float(np.max(loads)),
        "g_mean_savings": float(np.mean(sav)) if sav else float("nan"),
        "g_mean_salary": float(np.mean(sal)) if sal else float("nan"),
    }


FRIENDLY = {
    "log_amount": "Loan amount",
    "savings": "Savings",
    "salary": "Salary",
    "loan_to_savings": "Loan size vs savings",
    "loan_to_salary": "Loan size vs salary",
    "n_guarantors": "Number of guarantors",
    "g_prior_default_rate": "Guarantors who defaulted before",
    "g_load_asof_mean": "Guarantors' average load",
    "g_load_asof_max": "Most loaded guarantor",
    "g_mean_savings": "Guarantors' savings",
    "g_mean_salary": "Guarantors' salary",
}


def _shap(feats: dict, top: int = 6):
    """Per-feature SHAP contributions from the XGBoost model (native tree SHAP).
    Returns the strongest contributors, signed: 'up' raises risk, 'down' lowers it."""
    try:
        import pandas as pd
        import xgboost

        X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
        imp = MODEL.named_steps["imp"]
        clf = MODEL.named_steps["clf"]
        Xi = imp.transform(X)
        contribs = clf.get_booster().predict(
            xgboost.DMatrix(Xi, feature_names=FEATURES), pred_contribs=True
        )[0]
        out = []
        for i, f in enumerate(FEATURES):  # last entry is the bias term, skipped
            v = float(contribs[i])
            out.append(
                {"feature": f, "label": FRIENDLY.get(f, f), "value": round(v, 4),
                 "direction": "up" if v > 0 else "down"}
            )
        out.sort(key=lambda d: abs(d["value"]), reverse=True)
        return out[:top]
    except Exception:
        return []


def _flags_and_reasons(amount, savings, salary, disb, guarantor_ids, borrower_id):
    flags, reasons = [], []

    defaulters = [g for g in guarantor_ids if _member(g).get("ever_defaulted") == 1]
    if defaulters:
        flags.append(f"Backed by {len(defaulters)} guarantor(s) who have defaulted before")
        reasons.append({
            "label": "A guarantor has defaulted before", "direction": "up", "kind": "network",
            "detail": "One or more guarantors failed to repay a loan in the past: " + ", ".join(defaulters),
        })

    heavy = [(g, _member(g).get("loans_backed", 0)) for g in guarantor_ids
             if _member(g).get("loans_backed", 0) >= FLAG_TH["over_committed_loads"]]
    if heavy:
        flags.append("Over-committed guarantor: " + "; ".join(f"{g} backs {n} loans" for g, n in heavy))
        reasons.append({
            "label": "Over-committed guarantor", "direction": "up", "kind": "network",
            "detail": "A guarantor is already backing many loans, so their support is stretched thin.",
        })

    cdr = _member(borrower_id).get("community_default_rate", 0.0) if borrower_id else 0.0
    if cdr >= FLAG_TH["high_default_community"]:
        flags.append(f"Borrower in a high-default group ({cdr:.0%} default history)")
        reasons.append({
            "label": "High-default community", "direction": "up", "kind": "network",
            "detail": "The borrower sits in a guarantee group where many loans have gone bad.",
        })

    loan_to_savings = amount / ((savings or 0) + 1)
    if loan_to_savings >= 5:
        reasons.append({
            "label": "Loan large vs savings", "direction": "up", "kind": "individual",
            "detail": "The loan is large compared with the borrower's savings.",
        })
    elif loan_to_savings <= 1:
        reasons.append({
            "label": "Strong savings", "direction": "down", "kind": "individual",
            "detail": "The borrower's savings are healthy compared with the loan size.",
        })
    if not salary:
        reasons.append({
            "label": "No salary on file", "direction": "up", "kind": "individual",
            "detail": "No salary is recorded, so income is harder to confirm.",
        })

    if not flags:
        flags.append("No notable guarantor-network flags")
    return flags, reasons


def _heuristic(amount, savings, salary, guarantor_ids, borrower_id):
    """Transparent fallback used only if the model cannot be loaded."""
    p = 0.10
    if any(_member(g).get("ever_defaulted") == 1 for g in guarantor_ids):
        p += 0.35
    heavy = sum(1 for g in guarantor_ids if _member(g).get("loans_backed", 0) >= FLAG_TH["over_committed_loads"])
    p += min(0.20, 0.07 * heavy)
    cdr = _member(borrower_id).get("community_default_rate", 0.0) if borrower_id else 0.0
    if cdr >= FLAG_TH["high_default_community"]:
        p += 0.15
    if amount / ((savings or 0) + 1) >= 5:
        p += 0.15
    if not salary:
        p += 0.05
    return max(0.01, min(0.97, p))


def assess(amount, savings, salary, disb_date, guarantor_ids, borrower_id=None):
    disb = date.fromisoformat(disb_date) if disb_date else date.today()
    guarantor_ids = guarantor_ids or []
    savings = savings or 0.0

    flags, reasons = _flags_and_reasons(amount, savings, salary, disb, guarantor_ids, borrower_id)
    n_prior = sum(1 for g in guarantor_ids if _member(g).get("ever_defaulted") == 1)

    # New-member transparency: assess anyway, but say plainly what has no history yet.
    new_borrower = bool(borrower_id) and borrower_id not in MEMBERS
    new_guarantors = [g for g in guarantor_ids if g not in MEMBERS]
    new_members = ([borrower_id] if new_borrower else []) + new_guarantors
    if new_borrower or new_guarantors:
        bits = []
        if new_borrower:
            bits.append("the borrower is new to the system")
        if new_guarantors:
            bits.append(f"{len(new_guarantors)} guarantor(s) have no record yet (" + ", ".join(new_guarantors) + ")")
        flags[:] = [f for f in flags if f != "No notable guarantor-network flags"]
        flags.insert(0, "New member: " + "; ".join(bits) + ". Score uses the details you entered, "
                        "so treat it as a first estimate and confirm those details.")

    shap = []
    if MODEL is not None:
        import pandas as pd
        feats = _build_features(amount, savings, salary, disb, guarantor_ids)
        X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
        proba = float(MODEL.predict_proba(X)[0][1])
        shap = _shap(feats)
        source = "model"
    else:
        proba = _heuristic(amount, savings, salary, guarantor_ids, borrower_id)
        source = "heuristic"

    return {
        "risk_score": round(proba * 100),
        "band": _band(proba),
        "probability": round(proba, 4),
        "source": source,
        "reasons": reasons,
        "shap": shap,
        "flags": flags,
        "network": {
            "n_guarantors": len(guarantor_ids),
            "guarantors_with_prior_default": n_prior,
            "guarantor_ids": guarantor_ids,
            "new_members": new_members,
        },
    }
