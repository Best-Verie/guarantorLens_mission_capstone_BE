"""Reference dataset store.

The anonymised SACCO members, loans and guarantees are persisted in the database
(so admin uploads survive redeploys) and loaded read-only into memory for scoring and
the network views. The committed JSON artifacts are the *seed*: on first start the tables
are auto-seeded from them. If the database is unreachable, we fall back to reading the JSON
directly so the app still runs.
"""
import json
import os

from .db import Base, SessionLocal, engine
from .models import DatasetSeed, Guarantee, Loan, Member

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MEMBERS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_members.json")
LOANS_PATH = os.path.join(ARTIFACT_DIR, "guarantorlens_loans.json")

_MEMBER_COLS = ("member_id", "branch", "savings", "salary", "ever_defaulted", "loans_backed", "opening_date",
                "community_id", "community_default_rate")
_LOAN_COLS = ("loan_key", "borrower", "amount", "disb_date", "branch", "label", "outcome",
              "days_in_arrears", "payment_status", "troubled")

# loaded once, shared by scoring.py and network_data.py; cleared by reseed()
_CACHE = {"members": None, "loans": None}


def _mkey(m):
    return m.get("member_id") if m.get("member_id") is not None else m.get("member")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return []


def _member_row(m):
    mid = _mkey(m)
    return {c: (mid if c == "member_id" else m.get(c)) for c in _MEMBER_COLS}


def _loan_row(ln):
    return {c: (ln.get("borrower") or ln.get("member")) if c == "borrower" else ln.get(c)
            for c in _LOAN_COLS}


def _seed(db, members_data, loans_data):
    """Delete the existing reference data, then insert the given records, and record a
    DatasetSeed row (timestamp + counts) as proof. Returns the seed record dict."""
    # explicit delete of the previous dataset (guarantees first: FK to loans)
    deleted = (db.query(Guarantee).delete() + db.query(Loan).delete() + db.query(Member).delete())
    db.query(DatasetSeed).delete()
    db.commit()

    n_m = n_l = n_g = 0
    if members_data:
        mrows = [_member_row(m) for m in members_data if _mkey(m) is not None]
        db.bulk_insert_mappings(Member, mrows); n_m = len(mrows)
    if loans_data:
        lrows = [_loan_row(ln) for ln in loans_data if ln.get("loan_key")]
        db.bulk_insert_mappings(Loan, lrows); n_l = len(lrows)
        gts = [{"loan_key": ln["loan_key"], "guarantor": g}
               for ln in loans_data if ln.get("loan_key")
               for g in (ln.get("guarantors") or [])]
        if gts:
            db.bulk_insert_mappings(Guarantee, gts); n_g = len(gts)
    rec = DatasetSeed(members=n_m, loans=n_l, guarantees=n_g, deleted=deleted)
    db.add(rec)
    db.commit()
    return {"members": n_m, "loans": n_l, "guarantees": n_g, "deleted": deleted}


def ensure_seeded():
    """Create the tables and seed from the JSON artifacts if the members table is empty, or if it
    predates the guarantee-community columns (a one-time re-seed to backfill community_id)."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        empty = db.query(Member.member_id).first() is None
        try:
            needs_community = (not empty) and db.query(Member.member_id).filter(
                Member.community_id.isnot(None)).first() is None
        except Exception:
            needs_community = False   # column not migrated yet; main._ensure_columns adds it first
        if empty or needs_community:
            _seed(db, _read_json(MEMBERS_PATH), _read_json(LOANS_PATH))


def load_members() -> dict:
    """member_id -> member dict, read from the DB (JSON fallback). Column query for speed."""
    if _CACHE["members"] is not None:
        return _CACHE["members"]
    try:
        ensure_seeded()
        cols = [getattr(Member, c) for c in _MEMBER_COLS]
        with SessionLocal() as db:
            rows = db.query(*cols).all()
        data = {r[0]: dict(zip(_MEMBER_COLS, r)) for r in rows}
    except Exception:
        data = {_mkey(m): m for m in _read_json(MEMBERS_PATH) if _mkey(m) is not None}
    _CACHE["members"] = data
    return data


def load_loans() -> list:
    """List of loan dicts (each with a 'guarantors' list), from the DB (JSON fallback)."""
    if _CACHE["loans"] is not None:
        return _CACHE["loans"]
    try:
        ensure_seeded()
        cols = [getattr(Loan, c) for c in _LOAN_COLS]
        with SessionLocal() as db:
            lrows = db.query(*cols).all()
            grows = db.query(Guarantee.loan_key, Guarantee.guarantor).all()
        by_loan = {}
        for lk, g in grows:
            by_loan.setdefault(lk, []).append(g)
        data = []
        for r in lrows:
            d = dict(zip(_LOAN_COLS, r))
            d["guarantors"] = by_loan.get(r[0], [])
            data.append(d)
    except Exception:
        data = _read_json(LOANS_PATH)
    _CACHE["loans"] = data
    return data


def seed_info() -> dict | None:
    """The last seed record (timestamp + counts) - proof of the delete + re-seed."""
    try:
        with SessionLocal() as db:
            row = db.query(DatasetSeed).order_by(DatasetSeed.id.desc()).first()
        if row is not None:
            return {"seeded_at": row.seeded_at.isoformat() if row.seeded_at else None,
                    "members": row.members, "loans": row.loans,
                    "guarantees": row.guarantees, "deleted": row.deleted}
    except Exception:
        pass
    return None


def reseed(members_data, loans_data):
    """Admin upload: replace the reference tables from new JSON records, clear the cache."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        _seed(db, members_data, loans_data)
    _CACHE["members"] = None
    _CACHE["loans"] = None
