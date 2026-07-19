"""Acceptance testing — end-to-end user scenarios against the running API.

These follow the workflow a SACCO actually uses: a loan officer proposes and escalates
an application, a credit manager reviews and records a recommendation, and access control
is enforced. They exercise the deployed model through the API, not the logic in isolation.
"""
from tests.conftest import auth


def test_scenario_assess_escalate_recommend(client, officer_token, manager_token):
    # 1. Officer assesses a loan (saved as an application)
    created = client.post("/applications", json={
        "amount": 1_500_000, "savings": 200_000, "salary": 250_000, "guarantor_ids": []},
        headers=auth(officer_token))
    assert created.status_code == 201, created.text
    app_id = created.json()["id"]
    assert created.json()["status"] == "assessed"
    assert created.json()["band"] in ("Low", "Medium", "High")

    # 2. Officer escalates it to a credit manager
    esc = client.post(f"/applications/{app_id}/escalate", json={"note": "please review"},
                      headers=auth(officer_token))
    assert esc.status_code == 200, esc.text
    assert esc.json()["status"] == "escalated"

    # 3. Manager records a recommendation
    rec = client.post(f"/applications/{app_id}/recommendations",
                      json={"decision": "approve", "note": "guarantee is sufficient"},
                      headers=auth(manager_token))
    assert rec.status_code == 200, rec.text

    # 4. The application now carries the recommendation
    got = client.get(f"/applications/{app_id}", headers=auth(manager_token))
    assert got.status_code == 200
    assert len(got.json()["recommendations"]) >= 1


def test_scenario_officer_cannot_approve_own_case(client, officer_token):
    created = client.post("/applications", json={
        "amount": 900_000, "savings": 300_000, "guarantor_ids": []},
        headers=auth(officer_token))
    app_id = created.json()["id"]
    blocked = client.post(f"/applications/{app_id}/recommendations",
                          json={"decision": "approve"}, headers=auth(officer_token))
    assert blocked.status_code == 403, blocked.text


def test_scenario_manager_sees_escalation_queue(client, officer_token, manager_token):
    created = client.post("/applications", json={
        "amount": 1_000_000, "savings": 150_000, "guarantor_ids": []},
        headers=auth(officer_token))
    app_id = created.json()["id"]
    client.post(f"/applications/{app_id}/escalate", json={"note": "review"}, headers=auth(officer_token))
    queue = client.get("/applications?escalated=true", headers=auth(manager_token))
    assert queue.status_code == 200
    assert any(a["id"] == app_id for a in queue.json())
