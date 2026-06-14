"""Database engine/session. Uses DATABASE_URL (Postgres in prod), SQLite locally."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Read the connection string from the environment. No credentials are hardcoded.
# Falls back to a local SQLite file so the app runs locally without any setup.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./guarantorlens.db")
# Render/Heroku give "postgres://"; SQLAlchemy needs "postgresql+psycopg2://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()