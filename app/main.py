import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="GuarantorLens API",
    version="0.1.0",
    description="Network-aware, explainable loan-default risk for Umwalimu SACCO. "
                "Decision support for loan officers, not automatic approval.",
)

# CORS - allow the deployed frontend (set FRONTEND_ORIGIN, or * for the demo)
origins = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins] if origins != "*" else ["*"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)

@app.get("/health")
def health():
    return {"status": "ok"}