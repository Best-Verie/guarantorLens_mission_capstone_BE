"""SQLAlchemy ORM models."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

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
