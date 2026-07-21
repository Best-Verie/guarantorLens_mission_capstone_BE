"""Unit tests for the scoring logic (no HTTP, no auth).

These lock in the properties we rely on in the demo and in production:
band boundaries, display-score alignment and monotonicity, the guarantor-network
band overlay, and metamorphic sanity (more savings never raises risk; a bigger loan
never lowers it) for a new applicant with no guarantors, where the score is
deterministic and does not depend on the member table.
"""
import app.scoring as sc


# ---- band boundaries -------------------------------------------------------

def test_band_boundaries():
    m, h = sc.BANDS["medium"], sc.BANDS["high"]
    assert sc._band(m - 0.01) == "Low"
    assert sc._band(m) == "Medium"
    assert sc._band(h - 0.01) == "Medium"
    assert sc._band(h) == "High"


# ---- display score: alignment + monotonicity -------------------------------

def test_display_score_aligns_with_bands():
    """A High probability must never read as a low number, and vice-versa."""
    m, h = sc.BANDS["medium"], sc.BANDS["high"]
    assert sc._display_score(0.0) == sc._MIN_DISPLAY_SCORE   # display floor: never a literal 0/100
    assert sc._display_score(m / 2) < 40            # deep Low
    assert 40 <= sc._display_score((m + h) / 2) < 70  # mid Medium
    assert sc._display_score(h) >= 70               # High starts at 70
    assert sc._display_score(0.99) >= 90


def test_display_score_is_monotonic():
    scores = [sc._display_score(p / 100) for p in range(0, 101)]
    assert scores == sorted(scores)                 # never decreases as p rises
    assert 0 <= min(scores) and max(scores) <= 100


def test_display_never_zero():
    """The tool must never show a literal 0/100 or 0% - no model claims zero default risk."""
    assert sc._display_score(0.0) >= sc._MIN_DISPLAY_SCORE >= 1
    # even an all-but-impossible loan floors at the minimum display score
    assert min(sc._display_score(p / 1000) for p in range(0, 5)) >= sc._MIN_DISPLAY_SCORE


# ---- guarantor-network band overlay ----------------------------------------

def test_two_defaulter_backers_escalate_to_high():
    """Two guarantors who have defaulted before push the band to High by rule."""
    defaulters = [mid for mid, m in sc.MEMBERS.items() if m.get("ever_defaulted") == 1]
    if len(defaulters) < 2:
        import pytest
        pytest.skip("member table has fewer than two known defaulters")
    assert sc.adjust_band("Low", defaulters[:2]) == "High"


def test_no_guarantors_never_escalates():
    assert sc.adjust_band("Low", []) == "Low"
    assert sc.adjust_band("Medium", []) == "Medium"


# ---- metamorphic sanity (new applicant, no guarantors) ---------------------

NEW = dict(salary=200000, disb_date="2023-01-15", guarantor_ids=[], borrower_id=None)


def test_more_savings_does_not_raise_risk():
    scores = [sc.assess(amount=3_000_000, savings=s, **NEW)["risk_score"]
              for s in (50_000, 500_000, 2_000_000, 8_000_000)]
    assert all(b <= a for a, b in zip(scores, scores[1:])), scores  # non-increasing


def test_bigger_loan_does_not_lower_risk():
    scores = [sc.assess(amount=a, savings=150_000, **NEW)["risk_score"]
              for a in (300_000, 1_000_000, 4_000_000, 8_000_000)]
    assert all(b >= a for a, b in zip(scores, scores[1:])), scores  # non-decreasing


# ---- assess() contract -----------------------------------------------------

def test_assess_returns_expected_shape():
    r = sc.assess(amount=1_000_000, savings=200_000, salary=250_000,
                  disb_date="2023-03-01", guarantor_ids=[], borrower_id=None)
    for key in ("risk_score", "band", "probability", "source", "flags",
                "reasons", "recommendations", "brief", "network"):
        assert key in r, f"missing {key}"
    assert r["band"] in ("Low", "Medium", "High")
    assert 0 <= r["risk_score"] <= 100
    assert 0.0 <= r["probability"] <= 1.0
