"""Audit-trail verification for lending decisions and overrides.

GuarantorLens is decision-support: it never makes a binding decision, but every
human action on an application must leave an attributable, append-only record of
who did what and when. The recommendation rows and the application snapshot form
that audit trail. These tests assert the trail exists and cannot be quietly changed.
"""
from tests.conftest import auth


def _new_application(client, token, amount=1_000_000):
    r = client.post("/applications",
                    json={"amount": amount, "savings": 200_000, "salary": 250_000, "guarantor_ids": []},
                    headers=auth(token))
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_assessment_snapshot_is_persisted_with_source(client, officer_token):
    """Every application stores the score, band and model source at decision time."""
    app_id = _new_application(client, officer_token)
    r = client.get(f"/applications/{app_id}", headers=auth(officer_token))
    a = r.json()
    assert a["risk_score"] is not None and a["band"] in ("Low", "Medium", "High")
    assert a["source"] in ("model", "heuristic")     # the basis of the decision is recorded
    assert a["created_by_name"]                       # who created it
    assert a["created_at"]                            # when


def test_recommendation_is_attributable(client, officer_token, manager_token):
    """A manager's decision records author name, role, decision and timestamp."""
    app_id = _new_application(client, officer_token)
    r = client.post(f"/applications/{app_id}/recommendations",
                    json={"decision": "decline", "note": "guarantor over-committed"},
                    headers=auth(manager_token))
    assert r.status_code == 200, r.text
    recs = r.json()["recommendations"]
    assert len(recs) >= 1
    last = recs[-1]
    assert last["decision"] == "decline"
    assert last["author_role"] == "credit_manager"
    assert last["author_name"]
    assert last["created_at"]


def test_officer_cannot_record_a_decision(client, officer_token):
    """Separation of duties: a loan officer proposes but cannot record the decision."""
    app_id = _new_application(client, officer_token)
    r = client.post(f"/applications/{app_id}/recommendations",
                    json={"decision": "approve", "note": "self-approve attempt"},
                    headers=auth(officer_token))
    assert r.status_code == 403, r.text


def test_escalation_override_is_recorded(client, officer_token):
    """Escalating an application (an override of the branch outcome) is persisted with its note."""
    app_id = _new_application(client, officer_token)
    r = client.post(f"/applications/{app_id}/escalate",
                    json={"note": "aggregate guarantor exposure too high"},
                    headers=auth(officer_token))
    assert r.status_code == 200, r.text
    a = client.get(f"/applications/{app_id}", headers=auth(officer_token)).json()
    assert a["status"] == "escalated"
    assert a["escalation_note"] == "aggregate guarantor exposure too high"


def test_audit_trail_is_append_only(client, officer_token, manager_token):
    """A second decision adds a row; the first record stays unchanged (immutable, ordered)."""
    app_id = _new_application(client, officer_token)
    client.post(f"/applications/{app_id}/recommendations",
                json={"decision": "request_changes", "note": "need payslip"},
                headers=auth(manager_token))
    r2 = client.post(f"/applications/{app_id}/recommendations",
                     json={"decision": "approve", "note": "payslip received"},
                     headers=auth(manager_token))
    recs = r2.json()["recommendations"]
    assert len(recs) >= 2
    # the earlier decision is preserved verbatim and stays first in time order
    assert recs[0]["decision"] == "request_changes"
    assert recs[-1]["decision"] == "approve"
    assert recs[0]["created_at"] <= recs[-1]["created_at"]


def test_no_endpoint_edits_or_deletes_a_recommendation(client, officer_token, manager_token):
    """The trail only grows: there is no route to mutate or remove a recorded decision."""
    app_id = _new_application(client, officer_token)
    client.post(f"/applications/{app_id}/recommendations",
                json={"decision": "approve", "note": "ok"}, headers=auth(manager_token))
    rec_id = client.get(f"/applications/{app_id}",
                        headers=auth(manager_token)).json()["recommendations"][0]["id"]
    # neither PUT/PATCH nor DELETE on a recommendation should be a real, allowed route
    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)(f"/applications/{app_id}/recommendations/{rec_id}",
                                       headers=auth(manager_token))
        assert resp.status_code in (404, 405), f"{method} unexpectedly allowed: {resp.status_code}"


# --- dedicated append-only audit-log tests ----------------------------------------

def _audit(client, app_id, token):
    r = client.get(f"/applications/{app_id}/audit", headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def test_assessment_writes_an_audit_entry(client, officer_token, manager_token):
    """Creating an application logs one 'assess' entry attributed to the officer."""
    app_id = _new_application(client, officer_token)
    log = _audit(client, app_id, manager_token)
    assert [e["action"] for e in log] == ["assess"]
    e = log[0]
    assert e["actor_role"] == "loan_officer" and e["actor_name"]
    assert e["created_at"] and e["detail"]["band"] in ("Low", "Medium", "High")


def test_every_decision_and_override_is_logged_in_order(client, officer_token, manager_token):
    """assess -> escalate -> recommend each append exactly one immutable, ordered row."""
    app_id = _new_application(client, officer_token)
    client.post(f"/applications/{app_id}/escalate",
                json={"note": "exposure too high"}, headers=auth(officer_token))
    client.post(f"/applications/{app_id}/recommendations",
                json={"decision": "decline", "note": "over-committed"}, headers=auth(manager_token))
    log = _audit(client, app_id, manager_token)
    assert [e["action"] for e in log] == ["assess", "escalate", "recommend"]
    # the override and the decision captured who did what
    assert log[1]["action"] == "escalate" and log[1]["detail"]["note"] == "exposure too high"
    assert log[2]["actor_role"] == "credit_manager" and log[2]["detail"]["decision"] == "decline"
    # timestamps are non-decreasing (append-only order)
    assert log[0]["created_at"] <= log[1]["created_at"] <= log[2]["created_at"]


def test_whatif_override_is_captured_in_the_log(client, officer_token, manager_token):
    """A guarantor what-if override applied at assess time is recorded in the entry detail."""
    r = client.post("/applications",
                    json={"amount": 1_000_000, "savings": 200_000, "salary": 250_000,
                          "guarantor_ids": ["M1"],
                          "guarantor_overrides": {"M1": {"loans_backed": 9}}},
                    headers=auth(officer_token))
    assert r.status_code == 201, r.text
    app_id = r.json()["id"]
    log = _audit(client, app_id, manager_token)
    assert log[0]["detail"]["guarantor_overrides"] == {"M1": {"loans_backed": 9}}


def test_audit_log_is_append_only_no_write_route(client, officer_token, manager_token):
    """The log only grows: it has no create/edit/delete route of its own."""
    app_id = _new_application(client, officer_token)
    first = len(_audit(client, app_id, manager_token))
    client.post(f"/applications/{app_id}/recommendations",
                json={"decision": "approve", "note": "ok"}, headers=auth(manager_token))
    assert len(_audit(client, app_id, manager_token)) == first + 1
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(f"/applications/{app_id}/audit", headers=auth(manager_token))
        assert resp.status_code in (404, 405), f"{method} on audit unexpectedly allowed"
