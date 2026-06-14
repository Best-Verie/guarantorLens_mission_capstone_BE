"""Pydantic request/response models (shape the Swagger docs)."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# --- auth -------------------------------------------------------------------

ROLES = {"loan_officer", "credit_staff", "branch_manager"}


def _normalize_email(value: str) -> str:
    value = value.strip().lower()
    local, _, domain = value.partition("@")
    if not local or "." not in domain:
        raise ValueError("Enter a valid email address.")
    return value


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120, example="Beatrice Uwase")
    email: str = Field(..., example="b.uwase@umwalimusacco.rw")
    role: str = Field("loan_officer", example="loan_officer")
    password: str = Field(..., min_length=8, max_length=128, example="a-strong-password")

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _normalize_email(v)

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}")
        return v


class LoginRequest(BaseModel):
    email: str = Field(..., example="b.uwase@umwalimusacco.rw")
    password: str = Field(..., example="a-strong-password")

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return v.strip().lower()


class UserOut(BaseModel):
    id: int
    full_name: str
    email: str
    role: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., example="b.uwase@umwalimusacco.rw")

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return v.strip().lower()


class ForgotPasswordResponse(BaseModel):
    message: str
    # Only populated when DEBUG is on, so the flow is testable without email set up.
    reset_token: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., example="paste-the-token-from-the-email-link")
    new_password: str = Field(..., min_length=8, max_length=128, example="a-new-strong-password")


class MessageResponse(BaseModel):
    message: str


# --- risk assessment --------------------------------------------------------


class AssessRequest(BaseModel):
    borrower_id: Optional[str] = Field(None, example="Gasabo-335",
                                       description="Member id, used for the borrower's community history")
    amount: float = Field(..., example=1323000, description="Disbursement amount (RWF)")
    savings: float = Field(0, example=32036, description="Savings balance (RWF)")
    salary: Optional[float] = Field(None, example=91617, description="Monthly salary if on file")
    disbursement_date: Optional[str] = Field(None, example="2023-02-21",
                                             description="YYYY-MM-DD; defaults to today")
    guarantor_ids: List[str] = Field(default_factory=list,
                                     example=["Gasabo-189", "Gasabo-664", "Gasabo-366"])


class Reason(BaseModel):
    label: str
    direction: str           # "up" raises risk, "down" lowers risk
    detail: str
    kind: str                # "individual" | "network"


class NetworkInfo(BaseModel):
    n_guarantors: int
    guarantors_with_prior_default: int
    guarantor_ids: List[str]


class AssessResponse(BaseModel):
    risk_score: int          # 0-100
    band: str                # "Low" | "Medium" | "High"
    probability: float
    source: str              # "model" | "heuristic"
    reasons: List[Reason]
    flags: List[str]         # plain-language guarantor-network flags
    network: NetworkInfo


class MemberProfile(BaseModel):
    member_id: str
    branch: Optional[str] = None
    savings: Optional[float] = None
    salary: Optional[float] = None
    ever_defaulted: bool
    default_date: Optional[str] = None
    loans_backed: int            # guarantees this member has given (out-degree)
    total_connections: int       # degree in the guarantee graph
    community_default_rate: float


class LoanRef(BaseModel):
    loan_key: str
    amount: float
    disb_date: Optional[str] = None
    outcome: str
    guarantors: List[str] = []


class BackedLoan(BaseModel):
    loan_key: str
    borrower: str
    outcome: str


class NetNode(BaseModel):
    id: str
    role: str            # "self" | "backer" | "backed"
    ever_defaulted: bool
    loans_backed: int


class NetEdge(BaseModel):
    source: str          # guarantor
    target: str          # borrower


class MemberNetwork(BaseModel):
    nodes: List[NetNode]
    edges: List[NetEdge]


class MemberDetail(MemberProfile):
    loans: List[LoanRef] = []                 # loans where this member is the borrower
    backers: List[str] = []                   # members who guarantee this member's loans
    guarantees_given: List[BackedLoan] = []   # loans this member guarantees
    network: MemberNetwork


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
