"""Model-failure and fallback tests.

The service must never go dark if the model bundle fails to load or cannot score
(the usual cause is a scikit-learn / xgboost version mismatch on the host). In that
case scoring falls back to a transparent leak-free heuristic and every response is
tagged source="heuristic" so the failure is visible, not silent.
"""
import pytest

from app import scoring
from tests.conftest import auth


def test_load_falls_back_when_bundle_is_missing(monkeypatch):
    """If the joblib bundle cannot be read, _load() returns the heuristic path."""
    monkeypatch.setattr(scoring, "MODEL_PATH", "/no/such/bundle.joblib")
    loaded = scoring._load()
    model, features, source = loaded[0], loaded[1], loaded[7]
    assert model is None
    assert source == "heuristic"


def test_score_loan_without_model_returns_valid_probability(monkeypatch):
    """With MODEL=None the fast scorer still returns a usable probability and band."""
    monkeypatch.setattr(scoring, "MODEL", None)
    p, band = scoring.score_loan(3_000_000, 100_000, 200_000, "2023-06-01", [])
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    assert band in ("Low", "Medium", "High")


def test_assess_falls_back_when_model_raises(monkeypatch):
    """A model that loads but throws on predict_proba must degrade, not 500."""
    class _Broken:
        def predict_proba(self, X):
            raise RuntimeError("simulated version mismatch")
    monkeypatch.setattr(scoring, "MODEL", _Broken())
    out = scoring.assess(5_000_000, 60_000, 180_000, None, [])
    assert out["source"] == "heuristic"
    assert out["band"] in ("Low", "Medium", "High")
    assert 0 <= out["risk_score"] <= 100


def test_api_assessment_degrades_to_heuristic(client, officer_token, monkeypatch):
    """End to end: with no model loaded, /assess-risk still answers and says so."""
    monkeypatch.setattr(scoring, "MODEL", None)
    r = client.post("/assess-risk",
                    json={"amount": 5_000_000, "savings": 60_000, "salary": 180_000},
                    headers=auth(officer_token))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["source"] == "heuristic"
    assert data["band"] in ("Low", "Medium", "High")
    assert 0 <= data["risk_score"] <= 100


def test_model_info_reports_the_active_source(client, manager_token, monkeypatch):
    """The admin model card must state whether the real model or the fallback is live."""
    monkeypatch.setattr(scoring, "MODEL", None)
    monkeypatch.setattr(scoring, "_LOAD_SOURCE", "heuristic")
    info = scoring.model_info()
    assert info["source"] == "heuristic"
    assert info["loaded"] is False
