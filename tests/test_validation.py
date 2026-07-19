"""Validation testing — does the model/system meet its specified requirements?

Covers: the deployed model beats the baseline and meets its metric spec; the displayed
score never contradicts the band; and the API rejects invalid input.
"""
import os
import joblib
import pytest

import app.scoring as sc
from tests.conftest import auth

BUNDLE = os.path.join(os.path.dirname(sc.__file__), "artifacts", "guarantorlens_serving.joblib")


def test_deployed_model_beats_baseline_and_meets_spec():
    m = joblib.load(BUNDLE)["metrics"]
    # PR-AUC must be far above the base rate, and ROC-AUC strong.
    assert m["pr_auc"] > 10 * m["pr_baseline"], m
    assert m["pr_auc"] >= 0.45, m
    assert m["roc_auc"] >= 0.85, m


def test_bands_are_ordered():
    # calibrated band thresholds must be increasing (Low < Medium < High)
    assert 0 < sc.BANDS["medium"] < sc.BANDS["high"] < 1


def test_display_score_never_contradicts_band():
    floor = {"Low": 0, "Medium": 40, "High": 70}
    for p in [i / 1000 for i in range(0, 1001)]:
        band = sc._band(p)
        assert sc._display_for_band(p, band) >= floor[band]


def test_api_rejects_zero_amount(client, officer_token):
    r = client.post("/applications", json={"amount": 0, "guarantor_ids": []},
                    headers=auth(officer_token))
    assert r.status_code == 400, r.text


def test_api_rejects_missing_amount(client, officer_token):
    r = client.post("/assess-risk", json={"savings": 100000}, headers=auth(officer_token))
    assert r.status_code == 422, r.text  # pydantic validation
