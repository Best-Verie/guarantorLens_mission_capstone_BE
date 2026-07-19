"""SQLAlchemy ORM models."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    role = Column(String(40), nullable=False, default="loan_officer")
    hashed_password = Column(String(255), nullable=False)

    # Password reset: we store only the hash of the emailed token.
    reset_token_hash = Column(String(64), nullable=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class Application(Base):
    """A loan assessment saved as a reviewable application (the workflow unit)."""
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_by_name = Column(String(120), nullable=True)
    branch = Column(String(60), nullable=True)

    borrower_id = Column(String(60), nullable=True)     # null for a new applicant
    borrower_name = Column(String(120), nullable=True)
    amount = Column(Float, nullable=False)
    savings = Column(Float, nullable=True)
    salary = Column(Float, nullable=True)
    interest_rate = Column(Float, nullable=True)        # loan interest rate (%) if entered
    guarantor_ids = Column(Text, nullable=True)         # JSON list of member ids
    guarantor_overrides = Column(Text, nullable=True)   # JSON {id: {savings, salary, loans_backed}} what-if patches

    # assessment result snapshot
    risk_score = Column(Integer, nullable=True)
    band = Column(String(10), nullable=True)
    probability = Column(Float, nullable=True)
    reasons = Column(Text, nullable=True)               # JSON
    flags = Column(Text, nullable=True)                 # JSON
    source = Column(String(20), nullable=True)

    # workflow
    status = Column(String(20), nullable=False, default="assessed")  # assessed/escalated/recommended/closed
    escalation_note = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    recommendations = relationship("Recommendation", back_populates="application",
                                   cascade="all, delete-orphan", order_by="Recommendation.created_at")


class Recommendation(Base):
    """An officer's or manager's recommendation on an application (not a binding decision)."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False, index=True)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    author_name = Column(String(120), nullable=True)
    author_role = Column(String(40), nullable=True)
    decision = Column(String(30), nullable=False)       # approve / request_changes / decline
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    application = relationship("Application", back_populates="recommendations")


# --- Reference dataset (the anonymised SACCO member/loan/guarantee network) ------------
# Seeded from the JSON artifacts and used read-only for scoring and the network views.

class Member(Base):
    __tablename__ = "members"

    member_id = Column(String(60), primary_key=True, index=True)
    branch = Column(String(60), nullable=True)
    savings = Column(Float, nullable=True)
    salary = Column(Float, nullable=True)
    ever_defaulted = Column(Integer, nullable=True, default=0)
    loans_backed = Column(Integer, nullable=True, default=0)
    opening_date = Column(String(20), nullable=True)


class Loan(Base):
    __tablename__ = "loans"

    loan_key = Column(String(60), primary_key=True, index=True)
    borrower = Column(String(60), index=True, nullable=True)
    amount = Column(Float, nullable=True)
    disb_date = Column(String(20), nullable=True)
    branch = Column(String(60), nullable=True)
    label = Column(Integer, nullable=True)              # 1 = written off, 0 = normal
    outcome = Column(String(30), nullable=True)
    days_in_arrears = Column(Integer, nullable=True)
    payment_status = Column(String(30), nullable=True)
    troubled = Column(Integer, nullable=True)


class Guarantee(Base):
    __tablename__ = "guarantees"

    id = Column(Integer, primary_key=True, index=True)
    loan_key = Column(String(60), ForeignKey("loans.loan_key"), index=True, nullable=False)
    guarantor = Column(String(60), index=True, nullable=False)   # member_id of the guarantor


class DatasetSeed(Base):
    """One row recording the last seed: proves the reference data was deleted and re-seeded,
    with a timestamp and the resulting row counts."""
    __tablename__ = "dataset_seed"

    id = Column(Integer, primary_key=True, index=True)
    seeded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    members = Column(Integer, default=0)
    loans = Column(Integer, default=0)
    guarantees = Column(Integer, default=0)
    deleted = Column(Integer, default=0)      # rows removed from the previous dataset
