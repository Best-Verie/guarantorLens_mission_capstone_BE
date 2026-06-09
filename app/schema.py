"""Pydantic request/response models (shape the Swagger docs)."""
from typing import List, Optional
from pydantic import BaseModel, Field


class AssessRequest(BaseModel):
    member_id: Optional[int] = Field(None, example=1)
    amount: float = Field(..., example=2100000, description="Disbursement amount (RWF)")
    rate: float = Field(13, example=13, description="Interest rate (%)")
    savings: float = Field(0, example=395806, description="Savings balance (RWF)")
    salary: Optional[float] = Field(None, example=174312, description="Monthly salary if on file")
    disbursement_date: Optional[str] = Field(None, example="2023-09-19", description="YYYY-MM-DD")
    guarantor_ids: List[int] = Field(default_factory=list, example=[52, 24, 50])


class Reason(BaseModel):
    label: str
    direction: str           # "up" raises risk, "down" lowers risk
    detail: str
    kind: str                # "individual" | "network"


class NetworkInfo(BaseModel):
    n_guarantors: int
    guarantors_with_prior_default: int
    guarantor_ids: List[int]


class AssessResponse(BaseModel):
    risk_score: int          # 0-100
    band: str                # "Low" | "Medium" | "High"
    probability: float
    source: str              # "model" | "heuristic"
    reasons: List[Reason]
    network: NetworkInfo


class MemberOut(BaseModel):
    member_id: int
    branch: Optional[str] = None
    opening_date: Optional[str] = None
    n_loans: int
    n_repaid: int
    n_written_off: int
    backed_by: int           # in-degree (members who guarantee this one)
    guarantees_given: int    # out-degree
    neighbour_default_rate: float
