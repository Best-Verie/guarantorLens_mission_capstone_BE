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
    guarantor_ids = Column(Text, nullable=True)         # JSON list of member ids

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
