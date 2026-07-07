"""Loads the trained ("honest") model and the member/network table, and scores a loan.

This loader is BUNDLE-DRIVEN: it builds exactly the features listed in the model
bundle ("features"), so it works with either the master export or the simple
notebook's export without code changes. Any feature it cannot rebuild at serve
time is filled from the bundle's "medians" (the training medians). Risk bands
come from the bundle's "bands" (percentile-based) when present.

If the model cannot be loaded (missing file or library), the endpoint falls back
to a transparent heuristic so the API still responds.
"""
import json
import os
from bisect import bisect_left
from datetime import date

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_serving.joblib")
MEMBERS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_members.json")
LOANS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_loans.json")

_DEFAULT_BANDS = {"medium": 0.30, "high": 0.60}
_DEFAULT_FLAGS = {"over_committed_loads": 8, "high_default_community": 0.12}


def _mkey(m):
    """Member id under either the master ('member_id') or simple ('member') schema."""
    return m.get("member_id") if m.get("member_id") is not None else m.get("member")


def _load():
    members = {}
    try:
        with open(MEMBERS_PATH) as fh:
            members = {_mkey(m): m for m in json.load(fh) if _mkey(m) is not None}
    except Exception:
        members = {}

    # borrower -> sorted list of their loan disbursement dates, for as-of prior-loan counts
    loans_by_borrower = {}
    try:
        with open(LOANS_PATH) as fh:
            for ln in json.load(fh):
                bid = ln.get("borrower") if ln.get("borrower") is not None else ln.get("member")
                d = ln.get("disb_date") or ln.get("disb")
                if bid is None or not d:
                    continue
                try:
                    loans_by_borrower.setdefault(bid, []).append(date.fromisoformat(str(d)[:10]))
                except Exception:
                    pass
        for v in loans_by_borrower.values():
            v.sort()
    except Exception:
        loans_by_borrower = {}

    try:
        import joblib
        bundle = joblib.load(MODEL_PATH)
        # keep the small bundle metadata (name, metrics, trained_at, ...) for the model card
        meta = {k: v for k, v in bundle.items() if k not in ("model", "medians")}
        return (bundle["model"], bundle["features"], members, loans_by_borrower,
                bundle.get("bands") or _DEFAULT_BANDS,
                bundle.get("medians", {}),
                bundle.get("flag_thresholds", _DEFAULT_FLAGS), "model", meta)
    except Exception:
        return (None, None, members, loans_by_borrower,
                _DEFAULT_BANDS, {}, _DEFAULT_FLAGS, "heuristic", {})


MODEL, FEATURES, MEMBERS, LOANS_BY_BORROWER, BANDS, MEDIANS, FLAG_TH, _LOAD_SOURCE, MODEL_META = _load()


def reload():
    """Re-read the artifacts from disk and swap the in-memory model/tables. Used by the
    admin model-update endpoint after new artifacts are written."""
    global MODEL, FEATURES, MEMBERS, LOANS_BY_BORROWER, BANDS, MEDIANS, FLAG_TH, _LOAD_SOURCE, MODEL_META
    (MODEL, FEATURES, MEMBERS, LOANS_BY_BORROWER, BANDS, MEDIANS,
     FLAG_TH, _LOAD_SOURCE, MODEL_META) = _load()
    return _LOAD_SOURCE


def model_info():
    """Model-card data for the admin page: what is deployed right now."""
    meta = MODEL_META or {}
    raw = meta.get("metrics") or {k: v for k, v in meta.items()
                                  if k in ("roc_auc", "pr_auc", "pr_baseline")}
    metrics = {}
    for k, v in (raw or {}).items():
        try:
            metrics[k] = round(float(v), 4)
        except (TypeError, ValueError):
            metrics[k] = str(v)
    trained = meta.get("trained_at") or meta.get("created_at")
    return {
        "source": _LOAD_SOURCE,                       # "model" or "heuristic" fallback
        "loaded": MODEL is not None,
        "model_name": meta.get("model_name") or meta.get("name"),
        "trained_at": str(trained) if trained is not None else None,
        "n_features": len(FEATURES) if FEATURES else 0,
        "features": list(FEATURES) if FEATURES else [],
        "network_features": list(meta.get("network_features") or []),
        "bands": {k: float(v) for k, v in (BANDS or {}).items()},
        "flag_thresholds": {k: float(v) for k, v in (FLAG_TH or {}).items()},
        "metrics": metrics,
        "n_members": len(MEMBERS),
        "n_borrowers_with_loans": len(LOANS_BY_BORROWER),
    }


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


_BAND_ORDER = ["Low", "Medium", "High"]


def _bump(band: str, steps: int = 1) -> str:
    i = _BAND_ORDER.index(band) if band in _BAND_ORDER else 0
    return _BAND_ORDER[min(len(_BAND_ORDER) - 1, i + steps)]


def adjust_band(base_band, guarantor_ids, borrower_id=None):
    """Leak-free rule overlay: escalate the risk band on *concentrated* guarantor-network
    red flags. The model score is unchanged; this only raises the displayed band, because
    the model cannot reliably price guarantor commitment / defaults on this data.

    We escalate on concentrated signals, not any single link, because in a dense guarantor
    network almost every loan touches some over-committed or once-defaulted guarantor, so a
    single-link rule would flag nearly everything and tell the officer nothing.
      - whole backing group over-committed   -> +1 band (the guarantee itself is weak)
      - two or more backers defaulted before -> High (a serious cluster of bad backers)
    """
    band = base_band
    gs = guarantor_ids or []
    over = [g for g in gs if (_member(g).get("loans_backed") or 0) >= FLAG_TH["over_committed_loads"]]
    defaulters = [g for g in gs if _member(g).get("ever_defaulted") == 1]
    if gs and len(over) == len(gs):
        band = _bump(band, 1)
    if len(defaulters) >= 2:
        band = "High"
    return band


def score_loan(amount, savings, salary, disb_date, guarantor_ids, borrower_id=None):
    """Lightweight scorer for lists (e.g. the early-warning view): returns (probability, band).
    No flags or SHAP, so it stays fast over hundreds of loans."""
    try:
        disb = date.fromisoformat(str(disb_date)[:10]) if disb_date else date.today()
    except Exception:
        disb = date.today()
    guarantor_ids = guarantor_ids or []
    p = None
    if MODEL is not None:
        try:
            import pandas as pd
            feats = _build_features(amount, savings or 0.0, salary, disb, guarantor_ids, borrower_id)
            X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
            p = float(MODEL.predict_proba(X)[0][1])
        except Exception:
            p = None  # model could not score (e.g. version mismatch) -> fall back
    if p is None:
        p = _heuristic(amount, savings or 0.0, salary, guarantor_ids, borrower_id)
    return p, _band(p)


def score_many(items):
    """Batch scorer for lists (early-warning). One predict_proba call over all loans,
    so scoring hundreds of loans stays fast. items: list of dicts with amount, savings,
    salary, disb_date, guarantor_ids, borrower_id. Returns list of (probability, band)."""
    if not items:
        return []
    rows = []
    for it in items:
        try:
            disb = date.fromisoformat(str(it.get("disb_date"))[:10]) if it.get("disb_date") else date.today()
        except Exception:
            disb = date.today()
        feats = _build_features(it.get("amount", 0), it.get("savings") or 0.0, it.get("salary"),
                                disb, it.get("guarantor_ids") or [], it.get("borrower_id"))
        rows.append([feats[c] for c in FEATURES])
    probs = None
    if MODEL is not None:
        try:
            import pandas as pd
            probs = MODEL.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]
        except Exception:
            probs = None
    if probs is None:
        probs = [_heuristic(it.get("amount", 0), it.get("savings") or 0.0, it.get("salary"),
                            it.get("guarantor_ids") or [], it.get("borrower_id")) for it in items]
    return [(float(p), _band(float(p))) for p in probs]


def _prior_default(mid, disb) -> float:
    """1.0 if this member defaulted before `disb` (or has ever defaulted if no date)."""
    m = _member(mid)
    dt = m.get("default_date")
    if dt:
        try:
            return 1.0 if date.fromisoformat(dt) < disb else 0.0
        except Exception:
            pass
    return 1.0 if m.get("ever_defaulted") == 1 else 0.0


def _prior_loans(borrower_id, disb) -> int:
    """How many loans this borrower already had before `disb` (as-of, no leakage)."""
    if not borrower_id:
        return 0
    return sum(1 for d in LOANS_BY_BORROWER.get(borrower_id, []) if d < disb)


def _feat_value(name, amount, savings, salary, disb, guarantor_ids, borrower_id):
    """Compute one feature for a single loan. Returns None when it cannot be built
    (the caller then fills it from the training medians)."""
    import numpy as np

    g = guarantor_ids
    sav = [_member(x)["savings"] for x in g if _member(x).get("savings") is not None]
    sal = [_member(x)["salary"] for x in g if _member(x).get("salary") is not None]
    savings = savings or 0.0
    salary = salary or 0.0

    if name == "log_amount":
        return float(np.log1p(amount))
    if name == "savings":
        return float(savings)
    if name == "salary":
        return float(salary)
    if name == "loan_to_savings":
        return amount / (savings + 1)
    if name == "loan_to_salary":
        return amount / (salary + 1)
    if name == "n_guarantors":
        return len(g)
    if name == "g_mean_savings":
        return float(np.mean(sav)) if sav else None
    if name == "g_mean_salary":
        return float(np.mean(sal)) if sal else None
    if name == "g_sav_ratio":
        gms = float(np.mean(sav)) if sav else None
        return None if gms is None else gms / ((savings or 0.0) + 1)
    if name == "g_prior_default_rate":
        return float(np.mean([_prior_default(x, disb) for x in g])) if g else 0.0
    if name in ("g_load_asof_mean", "g_load_asof_max"):
        loads = [bisect_left(_guarantee_dates(x), disb) for x in g] or [0]
        return float(np.mean(loads)) if name.endswith("mean") else float(np.max(loads))
    if name == "b_prior_loans":
        return float(_prior_loans(borrower_id, disb))
    if name == "b_prior_writeoff":
        return _prior_default(borrower_id, disb) if borrower_id else 0.0
    if name == "b_account_age":
        m = _member(borrower_id)
        if m.get("account_age_days") is not None:
            return float(m["account_age_days"])
        return None  # no opening date stored -> impute from medians
    return None


def _build_features(amount, savings, salary, disb, guarantor_ids, borrower_id=None):
    """Build the bundle's feature vector for one loan, imputing gaps from medians."""
    out = {}
    for f in FEATURES:
        v = _feat_value(f, amount, savings, salary, disb, guarantor_ids, borrower_id)
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            v = float(MEDIANS.get(f, 0.0))
        out[f] = v
    return out


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
    "b_prior_loans": "Borrower's past loans",
    "b_prior_writeoff": "Borrower defaulted before",
    "b_account_age": "Borrower account age",
}


def _shap(feats: dict, top: int = 6):
    """Per-feature contributions from the XGBoost model (native tree SHAP).

    Handles the deployed shape (a CalibratedClassifierCV wrapping one imputer+XGB
    pipeline per CV fold): we average the tree contributions across folds. Also
    handles a bare pipeline or a bare classifier. Returns [] if none apply."""
    try:
        import numpy as np
        import pandas as pd
        import xgboost

        X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)

        def booster_contribs(estimator):
            # estimator may be a Pipeline (imputer + xgb) or a bare xgb classifier
            if hasattr(estimator, "steps"):
                clf = estimator[-1]
                Xi = estimator[:-1].transform(X)
            else:
                clf, Xi = estimator, X
            dm = xgboost.DMatrix(Xi, feature_names=list(FEATURES))
            return clf.get_booster().predict(dm, pred_contribs=True)[0]

        if hasattr(MODEL, "calibrated_classifiers_"):
            per_fold = []
            for cc in MODEL.calibrated_classifiers_:
                est = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
                per_fold.append(booster_contribs(est))
            contribs = np.mean(per_fold, axis=0)
        else:
            contribs = booster_contribs(MODEL)

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

    # Always explain the biggest individual drivers with the actual numbers, so the
    # score is never left unexplained (native SHAP is unavailable for this model).
    ratio = amount / ((savings or 0) + 1)
    if ratio >= 3:
        reasons.append({
            "label": f"Loan is {ratio:.0f}x the borrower's savings", "direction": "up", "kind": "individual",
            "detail": "A large loan relative to savings is the main thing pushing this score up.",
        })
    elif ratio >= 1.2:
        reasons.append({
            "label": f"Loan is {ratio:.1f}x the borrower's savings", "direction": "up", "kind": "individual",
            "detail": "The loan is somewhat larger than what the borrower has saved.",
        })
    else:
        reasons.append({
            "label": f"Savings cover the loan ({ratio:.1f}x)", "direction": "down", "kind": "individual",
            "detail": "Savings are healthy compared with the loan size, which lowers the score.",
        })

    if not salary:
        reasons.append({
            "label": "No salary on file", "direction": "up", "kind": "individual",
            "detail": "No salary is recorded, so income is harder to confirm.",
        })
    else:
        reasons.append({
            "label": "Salary on file", "direction": "down", "kind": "individual",
            "detail": "A recorded salary is evidence of income to repay.",
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
    proba = None
    source = "heuristic"
    if MODEL is not None:
        try:
            import pandas as pd
            feats = _build_features(amount, savings, salary, disb, guarantor_ids, borrower_id)
            X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
            proba = float(MODEL.predict_proba(X)[0][1])
            shap = _shap(feats)
            source = "model"
        except Exception:
            proba = None  # model could not score (e.g. version mismatch) -> fall back
    if proba is None:
        proba = _heuristic(amount, savings, salary, guarantor_ids, borrower_id)
        source = "heuristic"

    # Flag-adjusted band (leak-free rule overlay) - see adjust_band().
    model_band = _band(proba)
    band = adjust_band(model_band, guarantor_ids, borrower_id)
    if band != model_band:
        reasons.insert(0, {
            "label": f"Risk level raised to {band}", "direction": "up", "kind": "network",
            "detail": f"The model scored {round(proba * 100)}/100, but guarantor-network flags raise the "
                      "overall risk level. The model cannot price these reliably on this data, so they "
                      "escalate the band by rule.",
        })

    return {
        "risk_score": round(proba * 100),
        "band": band,
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
