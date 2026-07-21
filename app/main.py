import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import Base, engine
from . import models  # noqa: F401  (registers tables on Base)
from .auth import router as auth_router
from .risk import router as risk_router
from .members import router as members_router
from .insights import router as insights_router
from .applications import router as applications_router
from .admin import router as admin_router


def _ensure_columns():
    """Add columns introduced after the table was first created (create_all does not
    alter existing tables). Safe to run on every startup."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "applications" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("applications")}
    if "interest_rate" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE applications ADD COLUMN interest_rate FLOAT"))
    if "guarantor_overrides" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE applications ADD COLUMN guarantor_overrides TEXT"))
    for col in ("segment", "unusual"):
        if col not in existing:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE applications ADD COLUMN {col} TEXT"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create any missing tables (e.g. users) on startup, then patch in new columns.
    Base.metadata.create_all(bind=engine)
    try:
        _ensure_columns()
    except Exception:
        pass
    yield


app = FastAPI(
    title="GuarantorLens API",
    version="0.1.0",
    description="Network-aware, explainable loan-default risk for Umwalimu SACCO. "
                "Decision support for loan officers, not automatic approval.",
    lifespan=lifespan,
)

# CORS - allow the deployed frontend (set FRONTEND_ORIGIN, or * for the demo)
origins = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins] if origins != "*" else ["*"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)

app.include_router(auth_router)
app.include_router(risk_router)
app.include_router(members_router)
app.include_router(insights_router)
app.include_router(applications_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    return {"status": "ok"}
