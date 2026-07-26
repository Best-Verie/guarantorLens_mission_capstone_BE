"""Functional / system testing — the verified assessment scenarios.

Each case is a new applicant (borrower blank), guarantor-backed, and checks the risk
band and any expected guarantor-network flag. Bands and flags are stable; exact scores
may drift a point or two, so we assert the band, not the number.
"""
import pytest
import app.scoring as sc

CLEAN = ["Client32045", "Client41963"]
DEF = ["Client107745", "Client113217"]
OVER = ["Client120931"]

# (label, amount, savings, salary, rate, guarantors, expected_band, flag_substring)
CASES = [
    ("well-covered -> Low",        300000,  2500000, 400000, 13, CLEAN,               "Low",    None),
    ("moderate loan, higher rate -> Medium", 3000000, 600000, 350000, 14, CLEAN,      "Medium", None),
    ("over-extended -> High",      12000000, 30000,  150000, 14, CLEAN,               "High",   None),
    ("over-committed backer",      2000000,  300000, 300000, 14, OVER + ["Client32045"], "Medium", "over-committed"),
    ("two defaulter backers",      1500000,  200000, 250000, 14, DEF,                 "High",   "written off"),
]

_missing = [g for c in CASES for g in c[5] if g not in sc.MEMBERS]
pytestmark = pytest.mark.skipif(bool(_missing), reason=f"guarantor ids not in dataset: {_missing}")


@pytest.mark.parametrize("label,amount,savings,salary,rate,guars,band,flag", CASES)
def test_case_band_and_flag(label, amount, savings, salary, rate, guars, band, flag):
    r = sc.assess(amount, savings, salary, None, guars, interest_rate=rate)
    assert r["band"] == band, f"{label}: expected {band}, got {r['band']} ({r['risk_score']})"
    if flag:
        assert any(flag in f.lower() for f in r["flags"]), f"{label}: missing '{flag}' flag: {r['flags']}"


def test_interest_rate_lever_raises_risk():
    """Same loan and guarantors; a higher interest rate must not lower the risk band."""
    order = {"Low": 0, "Medium": 1, "High": 2}
    lo = sc.assess(5000000, 90000, 220000, None, CLEAN, interest_rate=13)
    hi = sc.assess(5000000, 90000, 220000, None, CLEAN, interest_rate=14)
    assert order[hi["band"]] >= order[lo["band"]], (lo["band"], hi["band"])
