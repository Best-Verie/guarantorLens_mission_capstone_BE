import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import Base, engine
from . import models  # noqa: F401  (registers tables on Base)
from .auth import router as auth_router
from .risk import router as risk_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create any missing tables (e.g. users) on startup.
    Base.metadata.create_all(bind=engine)
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


@app.get("/health")
def health():
    return {"status": "ok"}
