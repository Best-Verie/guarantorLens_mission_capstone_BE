"""Loads the trained ("honest") model and the member/network table, and scores a loan.

This loader is BUNDLE-DRIVEN: it builds exactly the features listed in the model
bundle ("features"), so it works with either the master export or the simple
notebook's export without code changes. Any feature it cannot rebuild at serve
time is filled from the bundle's "medians" (the training medians). Risk bands
come from the bundle's "bands" (percentile-based) when present.

If the model cannot be loaded (missing file or library), the endpoint falls back
to a transparent heuristic so the API still responds.
"""
import contextvars
import json
import os
import uuid
from bisect import bisect_left
from datetime import date

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_serving.joblib")
MEMBERS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_members.json")
LOANS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_loans.json")

_DEFAULT_BANDS = {"medium": 0.30, "high": 0.60}
_DEFAULT_FLAGS = {"over_committed_loads": 5, "high_default_community": 0.12}


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
    except Exception as e:
        # Surface WHY in the server logs (usually a library version mismatch on the host).
        import logging
        logging.getLogger("uvicorn.error").warning(
            "Model bundle failed to load from %s -> serving heuristic. Cause: %r", MODEL_PATH, e)
        return (None, None, members, loans_by_borrower,
                _DEFAULT_BANDS, {}, _DEFAULT_FLAGS, "heuristic", {})


MODEL, FEATURES, MEMBERS, LOANS_BY_BORROWER, BANDS, MEDIANS, FLAG_TH, _LOAD_SOURCE, MODEL_META = _load()


# --- loan histories for the as-of behavioural features ----------------------
# BORROWER_HISTORY: member -> sorted [(date, troubled)] for loans they took (as borrower)
# GUAR_BACKED:      member -> sorted [(date, troubled)] for loans they backed (as guarantor)
# "troubled" is written-off or 30+ days in arrears; falls back to the label if not present.
BORROWER_HISTORY: dict = {}
GUAR_BACKED: dict = {}


def _load_histories():
    bh, gb = {}, {}
    try:
        with open(LOANS_PATH) as fh:
            for ln in json.load(fh):
                d = ln.get("disb_date") or ln.get("disb")
                if not d:
                    continue
                try:
                    dt = date.fromisoformat(str(d)[:10])
                except Exception:
                    continue
                troubled = int(ln.get("troubled", ln.get("label", 0)) or 0)
                bid = ln.get("borrower") if ln.get("borrower") is not None else ln.get("member")
                if bid is not None:
                    bh.setdefault(bid, []).append((dt, troubled))
                for guar in (ln.get("guarantors") or []):
                    gb.setdefault(guar, []).append((dt, troubled))
        for v in bh.values():
            v.sort()
        for v in gb.values():
            v.sort()
    except Exception:
        pass
    return bh, gb


BORROWER_HISTORY, GUAR_BACKED = _load_histories()


# COMMUNITY: the borrower's guarantee cluster (connected component of the guarantee graph) and
# that cluster's historical default rate. Mirrors the notebook's `community_prior_default_rate`.
# At serve time "now" is after all recorded loans, so the as-of community rate is the full rate.
_COMMUNITY: dict = {}        # member_id -> community root
_COMMUNITY_RATE: dict = {}   # community root -> default rate over the cluster's loans


def _load_communities():
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    try:
        with open(LOANS_PATH) as fh:
            loans = json.load(fh)
    except Exception:
        return {}, {}
    for ln in loans:
        b = ln.get("borrower")
        if b is None:
            continue
        for g in (ln.get("guarantors") or []):
            union(b, g)
    agg = {}   # root -> [n_loans, n_bad]
    for ln in loans:
        b = ln.get("borrower")
        if b is None:
            continue
        root = find(b)
        cell = agg.setdefault(root, [0, 0])
        cell[0] += 1
        cell[1] += int(ln.get("label", 0) or 0)
    comm = {m: find(m) for m in parent}
    rate = {root: (bad / n if n else 0.0) for root, (n, bad) in agg.items()}
    return comm, rate


def _community_rate(borrower_id, guarantor_ids):
    """The default rate of the cluster this loan connects into (borrower's, else a guarantor's).
    None for a brand-new, unconnected member (then filled from the training median)."""
    root = _COMMUNITY.get(borrower_id) if borrower_id else None
    if root is None:
        for g in (guarantor_ids or []):
            if g in _COMMUNITY:
                root = _COMMUNITY[g]
                break
    return _COMMUNITY_RATE.get(root) if root is not None else None


_COMMUNITY, _COMMUNITY_RATE = _load_communities()


# --- opaque member ids for URLs ---------------------------------------------
# A member id like "Gasabo-001" is an (anonymised) account number: meaningful and
# enumerable. We never put it in a URL. Instead URLs carry a stable, opaque uid
# derived from the account number with a secret salt; the account number is still
# shown as text in the UI. Set MEMBER_UID_SALT in production so uids are not guessable.
_UID_NS = uuid.uuid5(uuid.NAMESPACE_URL,
                     "guarantorlens:" + os.getenv("MEMBER_UID_SALT", "gl-default-change-me"))
_UID_TO_MID: dict = {}


def member_uid(member_id):
    """Stable opaque URL id for a member (returns None for a missing id)."""
    return str(uuid.uuid5(_UID_NS, str(member_id))) if member_id is not None else None


def resolve_member_ref(ref):
    """Accept an opaque uid OR a raw account number; return the account number, or None."""
    if ref in MEMBERS:
        return ref
    if not _UID_TO_MID:
        _UID_TO_MID.update({member_uid(mid): mid for mid in MEMBERS})
    return _UID_TO_MID.get(ref)


def reload():
    """Re-read the artifacts from disk and swap the in-memory model/tables. Used by the
    admin model-update endpoint after new artifacts are written."""
    global MODEL, FEATURES, MEMBERS, LOANS_BY_BORROWER, BANDS, MEDIANS, FLAG_TH, _LOAD_SOURCE, MODEL_META
    global BORROWER_HISTORY, GUAR_BACKED, _COMMUNITY, _COMMUNITY_RATE
    (MODEL, FEATURES, MEMBERS, LOANS_BY_BORROWER, BANDS, MEDIANS,
     FLAG_TH, _LOAD_SOURCE, MODEL_META) = _load()
    BORROWER_HISTORY, GUAR_BACKED = _load_histories()
    _COMMUNITY, _COMMUNITY_RATE = _load_communities()
    _UID_TO_MID.clear()   # member table changed -> rebuild uid map lazily
    _CANDIDATE_CACHE.clear()   # rebuild the per-branch strong-guarantor pools from the new table
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


# What-if overrides: a per-call map {member_id: {savings, salary, loans_backed, ever_defaulted}}
# that temporarily patches a member's attributes so an officer can simulate "what if this
# guarantor had more savings / were not over-committed". Held in a context var so it is
# per-request safe (FastAPI runs sync endpoints in a threadpool). Set only inside assess().
_OVERRIDES: contextvars.ContextVar = contextvars.ContextVar("guarantor_overrides", default=None)

# only these attributes may be overridden, and each is coerced to the right type
_OVERRIDABLE = {
    "savings": float, "salary": float,
    "loans_backed": lambda v: int(float(v)), "ever_defaulted": lambda v: int(bool(v)),
}


def _clean_overrides(raw):
    if not raw:
        return None
    out = {}
    for mid, attrs in raw.items():
        if not isinstance(attrs, dict):
            continue
        clean = {}
        for k, coerce in _OVERRIDABLE.items():
            if attrs.get(k) is not None:
                try:
                    clean[k] = coerce(attrs[k])
                except (TypeError, ValueError):
                    pass
        if clean:
            out[mid] = clean
    return out or None


def _member(mid):
    m = MEMBERS.get(mid, {})
    ov = _OVERRIDES.get()
    if ov and mid in ov:
        return {**m, **ov[mid]}
    return m


def _guarantee_dates(mid):
    return sorted(date.fromisoformat(x) for x in _member(mid).get("guarantee_dates", []))


def _band(p: float) -> str:
    if p >= BANDS["high"]:
        return "High"
    if p >= BANDS["medium"]:
        return "Medium"
    return "Low"


def _display_score(p: float) -> int:
    """Map the calibrated probability onto an intuitive 0-100 risk score whose ranges match
    the bands: Low 0-39, Medium 40-69, High 70-100. This exists only for display, so an officer
    never sees "21/100 = High"; the true probability is still returned separately as `probability`.
    The map is monotonic in p, so a higher default probability always shows a higher score."""
    m, h = BANDS["medium"], BANDS["high"]
    if p < m:
        return int(round(39 * p / m)) if m > 0 else 0
    if p < h:
        return int(round(40 + 29 * (p - m) / (h - m))) if h > m else 55
    cap = max(h * 3.0, 0.60)     # probability at which the score saturates near 100
    return int(round(70 + 30 * min(1.0, (p - h) / (cap - h)))) if cap > h else 100


_BAND_ORDER = ["Low", "Medium", "High"]
_BAND_FLOOR = {"Low": 0, "Medium": 40, "High": 70}


def _display_for_band(p: float, band: str) -> int:
    """Display score that never contradicts the (possibly flag-escalated) band.

    The raw display score comes from the model probability, but adjust_band can raise the
    band on a leak-free rule (an over-committed or defaulter-backed guarantee) without
    changing the probability. Without this, a flagged loan reads e.g. 'High 46/100'. Since
    adjust_band only ever raises the band, we lift the score to that band's floor, so a
    flag-escalated loan reads High 70+, and the what-if delta moves the right way."""
    return max(_display_score(p), _BAND_FLOOR.get(band, 0))


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
            p = float(MODEL.predict_proba(X.values)[0][1])
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
            probs = MODEL.predict_proba(pd.DataFrame(rows, columns=FEATURES).values)[:, 1]
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


def _feat_value(name, amount, savings, salary, disb, guarantor_ids, borrower_id, interest_rate=None):
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
        return amount / (savings + 1) if savings else None   # missing savings -> median-fill (matches training)
    if name == "loan_to_salary":
        return amount / (salary + 1) if salary else None      # missing salary -> median-fill (matches training)
    if name == "n_guarantors":
        return len(g)
    if name == "g_mean_savings":
        return float(np.mean(sav)) if sav else None
    if name == "g_mean_salary":
        return float(np.mean(sal)) if sal else None
    if name == "g_sav_ratio":
        gms = float(np.mean(sav)) if sav else None
        return None if gms is None else gms / ((savings or 0.0) + 1)
    if name == "community_prior_default_rate":
        return _community_rate(borrower_id, guarantor_ids)
    if name == "g_prior_default_rate":
        return float(np.mean([_prior_default(x, disb) for x in g])) if g else 0.0
    if name == "g_prior_default_count":
        return float(sum(_prior_default(x, disb) for x in g)) if g else 0.0
    if name in ("g_load_asof_mean", "g_load_asof_max"):
        loads = [bisect_left(_guarantee_dates(x), disb) for x in g] or [0]
        return float(np.mean(loads)) if name.endswith("mean") else float(np.max(loads))
    if name in ("g_load_asof_sum", "g_loaded_count", "g_repeat_guarantor_rate"):
        loads = [bisect_left(_guarantee_dates(x), disb) for x in g] or [0]
        if name == "g_load_asof_sum":
            return float(np.sum(loads))
        loaded = sum(1 for x in loads if x > 0)
        if name == "g_loaded_count":
            return float(loaded)
        return float(loaded / len(loads)) if loads else 0.0
    if name in ("g_load_pressure", "g_default_pressure", "g_repeat_pressure"):
        log_amount = float(np.log1p(amount))
        loads = [bisect_left(_guarantee_dates(x), disb) for x in g] or [0]
        if name == "g_load_pressure":
            return log_amount * float(np.log1p(np.sum(loads)))
        if name == "g_default_pressure":
            prior_rate = float(np.mean([_prior_default(x, disb) for x in g])) if g else 0.0
            return log_amount * prior_rate
        loaded = sum(1 for x in loads if x > 0)
        repeat_rate = float(loaded / len(loads)) if loads else 0.0
        return log_amount * repeat_rate
    if name == "borrower_is_guarantor":
        return 1.0 if borrower_id and borrower_id in set(g) else 0.0
    if name == "b_prior_loans":
        return float(_prior_loans(borrower_id, disb))
    if name == "b_prior_writeoff":
        return _prior_default(borrower_id, disb) if borrower_id else 0.0
    if name == "b_account_age":
        m = _member(borrower_id)
        if m.get("account_age_days") is not None:
            return float(m["account_age_days"])
        return None  # no opening date stored -> impute from medians

    # --- as-of behavioural features (need the loan-history indexes) ---
    if name in ("b_prior_arrears_rate", "b_recent_loans", "time_since_last_loan"):
        prior = [(dt, tr) for dt, tr in BORROWER_HISTORY.get(borrower_id, []) if dt < disb] if borrower_id else []
        if name == "b_prior_arrears_rate":
            return float(np.mean([tr for _, tr in prior])) if prior else 0.0
        if name == "b_recent_loans":
            return float(sum(1 for dt, _ in prior if (disb - dt).days <= 730))
        return ((disb - max(dt for dt, _ in prior)).days / 365.25) if prior else None
    if name == "g_prior_arrears_rate":
        rates = []
        for x in g:
            backed = [tr for dt, tr in GUAR_BACKED.get(x, []) if dt < disb]
            rates.append(float(np.mean(backed)) if backed else 0.0)
        return float(np.mean(rates)) if g else 0.0
    if name == "account_age":
        m = _member(borrower_id)
        od = m.get("opening_date")
        if od:
            try:
                return max(0.0, (disb - date.fromisoformat(str(od)[:10])).days / 365.25)
            except Exception:
                pass
        if m.get("account_age") is not None:
            return float(m["account_age"])
        if m.get("account_age_days") is not None:
            return float(m["account_age_days"]) / 365.25
        # A borrower who is NOT in the system is genuinely new, so treat their tenure as ~0
        # rather than median-filling it (which would hand a first-time borrower ~6 years of
        # tenure they do not have, and unfairly LOWER their risk).
        if borrower_id and borrower_id not in MEMBERS:
            return 0.0
        return None  # borrower unknown / not specified -> impute from medians
    if name == "interest_rate":
        return float(interest_rate) if interest_rate is not None else None
    return None


def _build_features(amount, savings, salary, disb, guarantor_ids, borrower_id=None, interest_rate=None):
    """Build the bundle's feature vector for one loan, imputing gaps from medians."""
    out = {}
    for f in FEATURES:
        v = _feat_value(f, amount, savings, salary, disb, guarantor_ids, borrower_id, interest_rate)
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            v = float(MEDIANS.get(f, 0.0))
        out[f] = v
    return out


FRIENDLY = {
    "log_amount": "Loan amount",
    "savings": "Borrower's savings",
    "salary": "Borrower's salary",
    "loan_to_savings": "Loan vs borrower's savings",
    "loan_to_salary": "Loan vs borrower's salary",
    "n_guarantors": "Number of guarantors",
    "g_prior_default_rate": "Guarantors who defaulted before",
    "g_prior_default_count": "Prior-default guarantors",
    "g_load_asof_mean": "Guarantors' average load",
    "g_load_asof_max": "Most loaded guarantor",
    "g_load_asof_sum": "Total guarantor load",
    "g_loaded_count": "Previously active guarantors",
    "g_repeat_guarantor_rate": "Repeat-guarantor share",
    "g_load_pressure": "Network load pressure",
    "g_default_pressure": "Default-history pressure",
    "g_repeat_pressure": "Repeat-guarantor pressure",
    "borrower_is_guarantor": "Borrower also guarantees",
    "g_mean_savings": "Guarantors' savings",
    "g_mean_salary": "Guarantors' salary",
    "g_sav_ratio": "Guarantors' savings vs the loan",
    "community_prior_default_rate": "Guarantee network's default history",
    "b_prior_loans": "Borrower's past loans",
    "b_prior_writeoff": "Borrower defaulted before",
    "b_account_age": "Borrower account age",
    "b_prior_arrears_rate": "Borrower's past arrears",
    "b_recent_loans": "Borrower's recent loans",
    "time_since_last_loan": "Time since last loan",
    "g_prior_arrears_rate": "Guarantors' backing record",
    "interest_rate": "Interest rate",
    "account_age": "Borrower's account age",
}


def _driver_plain(feature, direction, feats):
    """A plain-language sentence explaining ONE model driver, so every bar in the drivers
    chart has a matching human explanation. Uses the actual feature value where it helps."""
    up = direction == "up"
    # With no guarantors attached, guarantor features are just median-fill artifacts -
    # never claim a guarantee that does not exist.
    if (feats.get("n_guarantors") or 0) == 0 and (feature.startswith("g_") or feature == "n_guarantors"):
        return "No guarantors are attached to this loan."
    lts = feats.get("loan_to_savings")
    lta = feats.get("loan_to_salary")
    sal = feats.get("salary")
    # drive the salary text by whether a salary is actually on file, not by the SHAP direction:
    # a missing salary is always "not on file"; a present-but-low salary raises risk.
    if not sal:
        salary_pair = ("No salary is on file, so income is unconfirmed.",
                       "No salary is on file, so income is unconfirmed.")
    else:
        salary_pair = ("The borrower's salary is modest for a loan this size, which adds some risk.",
                       "The borrower has a salary on file, which is evidence of income to repay.")
    D = {
        "log_amount": ("The loan amount is large, which raises risk.",
                       "The loan amount is modest, which lowers risk."),
        "savings": ("The borrower has little savings, which raises risk.",
                    "The borrower's savings are healthy relative to the loan, which lowers risk."),
        "salary": salary_pair,
        "loan_to_savings": (f"The loan is {lts:.0f}x the borrower's savings, a large loan relative to savings." if lts else
                            "The loan is large relative to the borrower's savings.",
                            "The loan is small relative to the borrower's savings, which lowers risk."),
        "loan_to_salary": (f"The loan is {lta:.0f}x the borrower's salary, a heavy load on income." if lta else
                           "The loan is large relative to the borrower's salary.",
                           "The loan is modest relative to the borrower's salary."),
        "account_age": ("The borrower's account is relatively new, so there is less history.",
                        "The borrower has been a member for a long time, which lowers risk."),
        "interest_rate": ("The loan's interest rate is on the higher side; the SACCO prices riskier loans higher.",
                          "The loan's interest rate is low, a sign of a lower-risk loan."),
        "b_prior_writeoff": ("The borrower has defaulted on a past loan, a strong risk signal.",
                             "The borrower has no past default on record."),
        "b_prior_arrears_rate": ("The borrower has fallen into arrears on past loans.",
                                 "The borrower has a clean repayment record."),
        "b_prior_loans": ("The borrower has taken several loans before.",
                          "The borrower has little past borrowing on record."),
        "b_recent_loans": ("The borrower has taken several loans recently, adding leverage.",
                           "The borrower has not borrowed much recently."),
        "time_since_last_loan": ("It has been a while since the borrower's last loan.",
                                 "The borrower borrowed recently."),
        "g_prior_arrears_rate": ("The guarantors have backed loans that fell into arrears before, which weakens the guarantee.",
                                 "The guarantors' past backing record is clean, which strengthens the guarantee."),
        "g_prior_default_rate": ("One or more guarantors have backed loans that defaulted.",
                                 "The guarantors have not backed a defaulted loan."),
        "g_mean_savings": ("The guarantors themselves hold little savings.",
                           "The guarantors hold savings, which strengthens the guarantee."),
        "g_mean_salary": ("The guarantors have little salary on file.",
                          "The guarantors have salaries on file, evidence they could step in."),
        "g_sav_ratio": ("The guarantors' savings are small relative to the loan.",
                        "The guarantors' savings are large relative to the loan, which strengthens the guarantee."),
        "n_guarantors": ("Few guarantors back this loan.",
                         "Several guarantors back this loan, which spreads the risk."),
        "community_prior_default_rate": ("The borrower's wider guarantee network has a high past-default rate, which raises risk.",
                                         "The borrower's wider guarantee network has a clean past-default record, which lowers risk."),
    }
    pair = D.get(feature)
    if not pair:
        label = FRIENDLY.get(feature, feature)
        return f"{label} { 'raises' if up else 'lowers' } the risk for this loan."
    return pair[0] if up else pair[1]


def _shap(feats: dict, top: int = 6):
    """Per-feature contributions from the XGBoost model (native tree SHAP).

    Handles the deployed shape (a CalibratedClassifierCV wrapping one imputer+XGB
    pipeline per CV fold): we average the tree contributions across folds. Also
    handles a bare pipeline or a bare classifier. Returns [] if none apply."""
    try:
        import numpy as np
        import pandas as pd
        import xgboost

        X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES).values  # plain array (no feature-name warning)

        def booster_contribs(estimator):
            # estimator may be a Pipeline (imputer + xgb) or a bare xgb classifier
            if hasattr(estimator, "steps"):
                clf = estimator[-1]
                Xi = X
                for _, step in estimator.steps[:-1]:
                    if hasattr(step, "transform"):
                        Xi = step.transform(Xi)
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

        # median-fill artifacts must not masquerade as real drivers:
        #  - with no guarantors, the guarantor-property features are just fill values
        #    (n_guarantors stays - "no guarantors" is a real factor);
        #  - with no salary on file, salary is stored as 0 and loan_to_salary is filled, so the
        #    model's contribution for them is an artifact, not a signal about this borrower. Showing
        #    "no salary on file" as a green "lowers risk" driver is misleading, so drop both.
        no_guar = (feats.get("n_guarantors") or 0) == 0
        no_sal = not feats.get("salary")
        skip = set()
        if no_guar:
            skip |= {"g_mean_savings", "g_mean_salary", "g_sav_ratio",
                     "g_prior_default_rate", "g_prior_arrears_rate"}
        if no_sal:
            skip |= {"salary", "loan_to_salary"}
        out = []
        for i, f in enumerate(FEATURES):  # last entry is the bias term, skipped
            if f in skip:
                continue
            v = float(contribs[i])
            direction = "up" if v > 0 else "down"
            out.append(
                {"feature": f, "label": FRIENDLY.get(f, f), "value": round(v, 4),
                 "direction": direction, "plain": _driver_plain(f, direction, feats)}
            )
        network_features = set((MODEL_META or {}).get("network_features") or [])
        for item in out:
            item["kind"] = "network" if item["feature"] in network_features else "individual"
        out.sort(key=lambda d: abs(d["value"]), reverse=True)
        top_items = out[:top]
        if network_features and not any(d["feature"] in network_features for d in top_items):
            network_ranked = [d for d in out if d["feature"] in network_features]
            if network_ranked:
                top_items = top_items[: max(0, top - 1)] + [network_ranked[0]]
        return top_items
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

    # Plain guarantor-finance reasons: how strong are the guarantors themselves?
    gsav = [_member(g).get("savings") for g in guarantor_ids if _member(g).get("savings") is not None]
    gsal = [_member(g).get("salary") for g in guarantor_ids if _member(g).get("salary") is not None]
    if gsav and amount:
        coverage = (sum(gsav) / len(gsav)) * len(guarantor_ids) / max(amount, 1.0)
        if coverage >= 0.5:
            reasons.append({
                "label": "Guarantors have strong savings", "direction": "down", "kind": "network",
                "detail": f"Together the guarantors' savings cover about {coverage:.0%} of the loan, "
                          "which strengthens the guarantee.",
            })
        elif coverage < 0.15:
            reasons.append({
                "label": "Guarantors have little savings", "direction": "up", "kind": "network",
                "detail": "The guarantors hold little savings themselves, so their guarantee is weaker.",
            })
    if gsal and salary:
        g_salary = sum(gsal) / len(gsal)
        if g_salary >= salary:
            reasons.append({
                "label": "Guarantors earn well", "direction": "down", "kind": "network",
                "detail": "On average the guarantors earn as much as or more than the borrower, "
                          "which strengthens the guarantee.",
            })
        elif g_salary < 0.6 * salary:
            reasons.append({
                "label": "Guarantors earn less than the borrower", "direction": "up", "kind": "network",
                "detail": "The guarantors earn noticeably less than the borrower, so they may struggle "
                          "to step in if the loan goes bad.",
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
            "label": f"Borrower's savings cover the loan ({ratio:.1f}x)", "direction": "down", "kind": "individual",
            "detail": "The borrower's savings are healthy compared with the loan size, which lowers the score.",
        })

    if not salary:
        reasons.append({
            "label": "Borrower's salary not on file", "direction": "up", "kind": "individual",
            "detail": "No salary is recorded for the borrower, so income is harder to confirm.",
        })
    else:
        reasons.append({
            "label": "Borrower's salary on file", "direction": "down", "kind": "individual",
            "detail": "A recorded salary for the borrower is evidence of income to repay.",
        })

    if not flags:
        flags.append("No notable guarantor-network flags")
    return flags, reasons


def _recommendations(amount, savings, salary, guarantor_ids, borrower_id, band):
    """Plain, actionable suggestions an officer could act on. Advice, not a decision."""
    recs = []
    if any(_member(g).get("ever_defaulted") == 1 for g in guarantor_ids):
        recs.append("Replace or add a guarantor: one of the current backers has defaulted before.")
    heavy = [g for g in guarantor_ids if (_member(g).get("loans_backed") or 0) >= FLAG_TH["over_committed_loads"]]
    if heavy:
        recs.append("Add a guarantor who is backing fewer loans, so the guarantee is not stretched thin.")
    gsav = [_member(g).get("savings") for g in guarantor_ids if _member(g).get("savings") is not None]
    if gsav and amount and (sum(gsav) / len(gsav)) * len(guarantor_ids) / max(amount, 1.0) < 0.15:
        recs.append("Add a guarantor with stronger savings to back the loan.")
    if amount / ((savings or 0) + 1) >= 3:
        recs.append("Consider a smaller loan, or ask the borrower to build up more savings first.")
    if not salary:
        recs.append("Confirm the borrower's income before approving; no salary is on file.")
    cdr = _member(borrower_id).get("community_default_rate", 0.0) if borrower_id else 0.0
    if cdr >= FLAG_TH["high_default_community"]:
        recs.append("Take a closer look at the borrower's guarantee group; it has a high default history.")
    if band == "High":
        recs.append("Escalate to a credit manager for a second review before deciding.")
    if not recs:
        recs.append("The profile looks sound. Proceed with the usual checks.")
    return recs


def _decision_brief(amount, savings, salary, guarantor_ids, borrower_id, band):
    """A short officer-style narrative that reads like a human wrote the memo. Template-based
    (deterministic, offline, defensible) - it just narrates the same facts the flags/reasons use,
    so the brief can never disagree with the score."""
    opener = {
        "High": "This loan looks high risk.",
        "Medium": "This loan is moderate risk.",
        "Low": "This loan looks low risk.",
    }.get(band, "This loan was assessed.")

    # borrower's own footing
    cover = (savings or 0) / max(amount, 1.0)
    if cover >= 2:
        borrower = f"The borrower is well covered, with savings about {cover:.1f} times the loan"
    elif cover >= 0.5:
        borrower = f"The borrower's savings cover roughly {cover:.0%} of the loan"
    else:
        borrower = "The borrower has thin savings against the loan"
    if borrower_id and _member(borrower_id).get("ever_defaulted") == 1:
        borrower += ", and they have defaulted on a past loan"
    elif not salary:
        borrower += ", and no salary is on file"
    borrower += "."

    # the guarantee
    gparts = []
    defaulters = [g for g in guarantor_ids if _member(g).get("ever_defaulted") == 1]
    heavy = [g for g in guarantor_ids if (_member(g).get("loans_backed") or 0) >= FLAG_TH["over_committed_loads"]]
    gsav = [_member(g).get("savings") for g in guarantor_ids if _member(g).get("savings") is not None]
    gcover = (sum(gsav) / len(gsav)) * len(guarantor_ids) / max(amount, 1.0) if gsav and amount else None
    if not guarantor_ids:
        guarantee = "No guarantors are attached to this loan."
    else:
        if defaulters:
            gparts.append(f"{len(defaulters)} of {len(guarantor_ids)} guarantors have defaulted before")
        if heavy:
            gparts.append(f"{len(heavy)} guarantor(s) are over-committed, backing many loans at once")
        if gcover is not None and gcover >= 0.5:
            gparts.append(f"together the guarantors' savings cover about {gcover:.0%} of the loan")
        elif gcover is not None and gcover < 0.15:
            gparts.append("the guarantors themselves hold little savings")
        if not gparts:
            gparts.append(f"the {len(guarantor_ids)} guarantor(s) raise no particular red flags")
        guarantee = "On the guarantee side, " + "; ".join(gparts) + "."

    closer = {
        "High": "Recommend a second review by a credit manager, and strengthening the guarantee before approval.",
        "Medium": "Proceed with care; a stronger guarantor or a smaller amount would lower the risk.",
        "Low": "Proceed with the usual checks.",
    }.get(band, "")
    return f"{opener} {borrower} {guarantee} {closer}".strip()


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


_CANDIDATE_CACHE: dict = {}   # branch -> [member_ids]


def _strong_candidates(branch=None, limit=40):
    """A pool of strong, available guarantors: clean record, not over-committed, real savings.
    Filtered to `branch` when given, because at this SACCO a member can only guarantee a
    borrower in their OWN branch. Cached per branch (depends only on the static member table)."""
    key = branch or "__all__"
    if key not in _CANDIDATE_CACHE:
        pool = [
            m for m in MEMBERS.values()
            if not m.get("ever_defaulted")
            and (m.get("loans_backed") or 0) < FLAG_TH["over_committed_loads"]
            and (m.get("savings") or 0) > 0
            and (branch is None or m.get("branch") == branch)
        ]
        pool.sort(key=lambda m: (m.get("savings") or 0), reverse=True)
        _CANDIDATE_CACHE[key] = [m["member_id"] for m in pool[:limit]]
    return _CANDIDATE_CACHE[key]


def _borrower_branch(borrower_id, guarantor_ids):
    """The branch a guarantor must belong to = the borrower's branch. If the borrower is new
    (no record), fall back to the branch of the guarantors already on the loan, since the
    same-branch rule means they are in the borrower's branch too."""
    b = _member(borrower_id).get("branch") if borrower_id else None
    if b:
        return b
    for g in guarantor_ids or []:
        gb = _member(g).get("branch")
        if gb:
            return gb
    return None


def _guarantor_weakness(g):
    """Higher = weaker guarantor, so we know which one to try replacing first."""
    m = _member(g)
    score = 0.0
    if m.get("ever_defaulted") == 1:
        score += 1000
    if (m.get("loans_backed") or 0) >= FLAG_TH["over_committed_loads"]:
        score += 500 + (m.get("loans_backed") or 0)
    score += 1.0 / ((m.get("savings") or 0) + 1)   # thinner savings -> weaker
    return score


def suggest_guarantors(amount, savings, salary, guarantor_ids, borrower_id=None, interest_rate=None, max_suggestions=3):
    """For a loan that scores Medium/High, search for a single guarantor change (swap the weakest
    current backer, or add one if there is room) that lowers the risk band, and return the best
    few. Turns the tool from a judge into an advisor: 'do this and the loan becomes bankable.'"""
    guarantor_ids = [g for g in (guarantor_ids or [])]
    base = assess(amount, savings, salary, None, guarantor_ids, borrower_id, interest_rate)
    current = {"score": base["risk_score"], "band": base["band"]}
    if base["band"] == "Low":
        return {"current": current, "suggestions": [], "branch": None,
                "message": "This loan is already low risk."}

    # A guarantor must be in the borrower's own branch, so only search there.
    branch = _borrower_branch(borrower_id, guarantor_ids)
    candidates = [c for c in _strong_candidates(branch=branch)
                  if c not in guarantor_ids and c != borrower_id]
    weakest = max(guarantor_ids, key=_guarantor_weakness) if guarantor_ids else None
    band_rank = {"Low": 0, "Medium": 1, "High": 2}
    results = []
    for cand in candidates:
        trials = []
        if weakest is not None:
            swapped = [c for c in guarantor_ids if c != weakest] + [cand]
            trials.append(("swap", weakest, swapped))
        if len(guarantor_ids) < 3:
            trials.append(("add", None, guarantor_ids + [cand]))
        for action, removed, gids in trials:
            r = assess(amount, savings, salary, None, gids, borrower_id, interest_rate)
            improved = (band_rank[r["band"]] < band_rank[base["band"]]) or (r["risk_score"] < base["risk_score"] - 3)
            if improved:
                cm = _member(cand)
                results.append({
                    "action": action, "remove": removed, "add": cand,
                    "new_score": r["risk_score"], "new_band": r["band"],
                    "delta": r["risk_score"] - base["risk_score"],
                    "add_savings": cm.get("savings"), "add_loans_backed": int(cm.get("loans_backed") or 0),
                    "add_branch": cm.get("branch"),
                    "why": ("stronger savings, clean record, backs few loans"),
                })
        if len(results) >= max_suggestions * 3:   # enough good options; stop scoring
            break
    # best first: biggest band drop, then biggest score drop
    results.sort(key=lambda x: (band_rank[x["new_band"]], x["delta"]))
    # de-dup by the added member, keep the strongest single move per candidate
    seen, unique = set(), []
    for r in results:
        if r["add"] in seen:
            continue
        seen.add(r["add"]); unique.append(r)
    msg = ""
    if not unique:
        msg = (f"No same-branch guarantor change moved the band; consider a smaller loan or more savings."
               if branch else "No single guarantor change moved the band; consider a smaller loan or more savings.")
    return {"current": current, "suggestions": unique[:max_suggestions],
            "weakest_current": weakest, "branch": branch, "message": msg}


def assess(amount, savings, salary, disb_date, guarantor_ids, borrower_id=None,
           interest_rate=None, guarantor_overrides=None):
    """Score a loan. `guarantor_overrides` powers the what-if: a map
    {member_id: {savings, salary, loans_backed, ever_defaulted}} that temporarily patches those
    guarantors' attributes for THIS call only, so an officer can simulate 'what if this guarantor
    had more savings, or were not over-committed'. It flows to every guarantor read via _member."""
    token = _OVERRIDES.set(_clean_overrides(guarantor_overrides))
    try:
        return _assess_core(amount, savings, salary, disb_date, guarantor_ids,
                            borrower_id, interest_rate)
    finally:
        _OVERRIDES.reset(token)


def _assess_core(amount, savings, salary, disb_date, guarantor_ids, borrower_id=None, interest_rate=None):
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
            feats = _build_features(amount, savings, salary, disb, guarantor_ids, borrower_id, interest_rate)
            X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
            proba = float(MODEL.predict_proba(X.values)[0][1])
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
            "detail": f"The model scored {_display_score(proba)}/100, but guarantor-network flags raise the "
                      "overall risk level. The model cannot price these reliably on this data, so they "
                      "escalate the band by rule.",
        })

    recommendations = _recommendations(amount, savings, salary, guarantor_ids, borrower_id, band)
    brief = _decision_brief(amount, savings, salary, guarantor_ids, borrower_id, band)

    return {
        "risk_score": _display_for_band(proba, band),
        "band": band,
        "probability": round(proba, 4),
        "source": source,
        "brief": brief,
        "reasons": reasons,
        "recommendations": recommendations,
        "shap": shap,
        "flags": flags,
        "network": {
            "n_guarantors": len(guarantor_ids),
            "guarantors_with_prior_default": n_prior,
            "guarantor_ids": guarantor_ids,
            "new_members": new_members,
        },
        # opaque url ids for the borrower + guarantors, so links never expose account numbers
        "uids": {r: member_uid(r) for r in ([borrower_id] + list(guarantor_ids)) if r},
    }
