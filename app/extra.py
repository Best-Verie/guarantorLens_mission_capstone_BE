"""Two extra per-application models, loaded from a joblib bundle fit in extra_models.ipynb:

  - borrower SEGMENT  (KMeans, unsupervised) - which borrower 'type' this application is closest to,
  - anomaly FLAG      (Isolation Forest)      - how unusual this application looks.

These are display-only context for the officer. The risk score itself stays the classifier's job.
Both degrade gracefully: if the bundle is missing or a lookup fails, the helpers return None and the
API simply omits the fields (same pattern as the rule-based fallback for the main model).
"""
import os
import numpy as np
import pandas as pd

_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "guarantorlens_extra.joblib")
try:
    import joblib
    _B = joblib.load(_PATH)
except Exception:
    _B = None

HAS_EXTRA = _B is not None


def _vec(amount, savings, salary, n_guar):
    ltsav = (amount or 0) / ((savings or 0) + 1)
    row = pd.DataFrame([[amount or 0, savings, salary, ltsav, n_guar or 0]], columns=_B["feats"])
    x = _B["imputer"].transform(row)
    x = np.clip(x, _B["winsor_lo"], _B["winsor_hi"])
    return _B["scaler"].transform(x)


def segment(amount, savings, salary, n_guar):
    """Closest borrower segment + that segment's historical write-off rate."""
    if _B is None:
        return None
    try:
        cl = int(_B["kmeans"].predict(_vec(amount, savings, salary, n_guar))[0])
        return {"id": cl,
                "description": _B["cluster_desc"].get(cl, f"segment {cl}"),
                "segment_write_off_rate": _B["cluster_bad_rate"].get(cl)}
    except Exception:
        return None


def anomaly(amount, savings, salary, n_guar):
    """Is this application unusual vs the book? (atypical, not necessarily risky.)"""
    if _B is None:
        return None
    try:
        s = float(-_B["iso"].score_samples(_vec(amount, savings, salary, n_guar))[0])
        return {"unusual": bool(s >= _B["anom_threshold"]), "score": round(s, 3)}
    except Exception:
        return None
