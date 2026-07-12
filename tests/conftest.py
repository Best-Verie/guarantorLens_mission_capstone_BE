"""Shared test fixtures.

Points the app at a throwaway SQLite file BEFORE the app is imported, so tests never
touch the real dev database, then exposes a TestClient plus ready-made officer and
manager auth tokens.
"""
import os
import tempfile

import pytest

# Must be set before any app module imports db.engine.
_TMP_DB = tempfile.mkstemp(suffix=".db")[1]
os.environ["DATABASE_URL"] = "sqlite:///" + _TMP_DB

from app.db import Base, engine        # noqa: E402
from app.main import app               # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def _make_user(client, email, role):
    """Register (or log in if already there) and return a Bearer token."""
    body = {"full_name": "Test User", "email": email, "role": role, "password": "test-pass-123"}
    r = client.post("/auth/register", json=body)
    if r.status_code == 409:
        r = client.post("/auth/login", json={"email": email, "password": "test-pass-123"})
    assert r.status_code in (200, 201), r.text
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def officer_token(client):
    return _make_user(client, "officer.test@sacco.rw", "loan_officer")


@pytest.fixture(scope="session")
def manager_token(client):
    return _make_user(client, "manager.test@sacco.rw", "credit_manager")


def auth(token):
    return {"Authorization": f"Bearer {token}"}
