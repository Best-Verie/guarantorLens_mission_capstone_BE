"""Integration tests: real HTTP against the FastAPI app with auth.

Covers the endpoints the demo relies on and the access-control rule that a loan
officer cannot record a manager's recommendation.
"""
from tests.conftest import auth


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_assess_risk_requires_auth(client):
    r = client.post("/assess-risk", json={"amount": 1_000_000})
    assert r.status_code in (401, 403)


def test_assess_risk_returns_band_and_score(client, officer_token):
    body = {"amount": 5_000_000, "savings": 60_000, "salary": 180_000,
            "interest_rate": 14, "guarantor_ids": []}
    r = client.post("/assess-risk", json=body, headers=auth(officer_token))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["band"] in ("Low", "Medium", "High")
    assert 0 <= data["risk_score"] <= 100


def test_well_covered_scores_lower_than_thin(client, officer_token):
    """Sanity through the API: a well-covered small loan should not score higher
    than a thin, large one."""
    h = auth(officer_token)
    low = client.post("/assess-risk", json={
        "amount": 300_000, "savings": 2_500_000, "salary": 400_000, "interest_rate": 13},
        headers=h).json()
    high = client.post("/assess-risk", json={
        "amount": 5_000_000, "savings": 60_000, "salary": 180_000, "interest_rate": 14},
        headers=h).json()
    assert low["risk_score"] <= high["risk_score"]


def test_officer_cannot_recommend(client, officer_token):
    """Negative / access-control test: an officer creates an application, then is
    blocked (403) from recording a credit-manager recommendation on it."""
    created = client.post("/applications", json={
        "amount": 1_000_000, "savings": 200_000, "salary": 250_000, "guarantor_ids": []},
        headers=auth(officer_token))
    assert created.status_code == 201, created.text
    app_id = created.json()["id"]

    r = client.post(f"/applications/{app_id}/recommendations",
                    json={"decision": "approve", "note": "looks fine"},
                    headers=auth(officer_token))
    assert r.status_code == 403, r.text


def test_manager_can_recommend(client, officer_token, manager_token):
    created = client.post("/applications", json={
        "amount": 800_000, "savings": 300_000, "salary": 250_000, "guarantor_ids": []},
        headers=auth(officer_token))
    app_id = created.json()["id"]
    r = client.post(f"/applications/{app_id}/recommendations",
                    json={"decision": "approve", "note": "approved by manager"},
                    headers=auth(manager_token))
    assert r.status_code == 200, r.text
