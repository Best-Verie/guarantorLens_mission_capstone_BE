"""Database engine/session. Uses DATABASE_URL (Postgres in prod), SQLite locally."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://guarantor_lens_user:frJWIoTqOWHQrQPe9pweBFG6Uz1KRYPB@dpg-d8k6g0kvikkc73bt89fg-a.oregon-postgres.render.com/guarantor_lens")
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