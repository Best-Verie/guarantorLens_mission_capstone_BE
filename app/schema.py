"""Pydantic request/response models (shape the Swagger docs)."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# --- auth -------------------------------------------------------------------

ROLES = {"loan_officer", "credit_manager", "admin"}


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


class ShapContribution(BaseModel):
    feature: str
    label: str               # friendly name
    value: float             # signed SHAP value (log-odds); >0 raises risk
    direction: str           # "up" | "down"


class AssessResponse(BaseModel):
    risk_score: int          # 0-100
    band: str                # "Low" | "Medium" | "High"
    probability: float
    source: str              # "model" | "heuristic"
    reasons: List[Reason]
    shap: List[ShapContribution] = []   # model-faithful per-feature contributions
    flags: List[str]         # plain-language guarantor-network flags
    network: NetworkInfo
    uids: dict = {}          # account number -> opaque url id (borrower + guarantors)


# --- insights --------------------------------------------------------------

class WatchlistItem(BaseModel):
    loan_key: str
    borrower: str
    borrower_uid: Optional[str] = None
    branch: Optional[str] = None
    amount: float
    days_in_arrears: int
    backed_by_defaulter: bool


class SuperGuarantor(BaseModel):
    member_id: str
    uid: Optional[str] = None
    branch: Optional[str] = None
    loans_backed: int
    ever_defaulted: bool
    bad_loans_backed: int


class CommunityStat(BaseModel):
    community_id: str
    branch: Optional[str] = None
    size: int
    default_rate: float


class EarlyWarningItem(BaseModel):
    loan_key: str
    borrower: str
    borrower_uid: Optional[str] = None
    branch: Optional[str] = None
    amount: float
    days_in_arrears: int
    risk_score: int
    band: str


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


class NetworkView(BaseModel):
    center: str
    nodes: List[NetNode]
    edges: List[NetEdge]


class MemberDetail(MemberProfile):
    uid: Optional[str] = None                 # opaque url id for this member
    uids: dict = {}                           # account number -> uid for every member linked on the page
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


# --- applications workflow ---------------------------------------------------

class ApplicationCreate(BaseModel):
    amount: float
    savings: float = 0.0
    salary: Optional[float] = None
    guarantor_ids: List[str] = []
    borrower_id: Optional[str] = None
    borrower_name: Optional[str] = None
    branch: Optional[str] = None


class RecommendationCreate(BaseModel):
    decision: str            # approve / request_changes / decline
    note: Optional[str] = None


class RecommendationOut(BaseModel):
    id: int
    author_name: Optional[str] = None
    author_role: Optional[str] = None
    decision: str
    note: Optional[str] = None
    created_at: str


class EscalateRequest(BaseModel):
    note: Optional[str] = None


class ApplicationListItem(BaseModel):
    id: int
    borrower: Optional[str] = None
    branch: Optional[str] = None
    amount: float
    risk_score: Optional[int] = None
    band: Optional[str] = None
    status: str
    created_by_name: Optional[str] = None
    created_at: str


class ApplicationOut(BaseModel):
    id: int
    created_by_name: Optional[str] = None
    branch: Optional[str] = None
    borrower_id: Optional[str] = None
    borrower_name: Optional[str] = None
    amount: float
    savings: Optional[float] = None
    salary: Optional[float] = None
    guarantor_ids: List[str] = []
    risk_score: Optional[int] = None
    band: Optional[str] = None
    probability: Optional[float] = None
    reasons: list = []
    flags: List[str] = []
    source: Optional[str] = None
    status: str
    escalation_note: Optional[str] = None
    created_at: str
    recommendations: List[RecommendationOut] = []


class ApplicationStats(BaseModel):
    my_open: int
    escalated: int
    total: int


# --- admin ------------------------------------------------------------------

class AdminUserOut(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    created_at: Optional[str] = None
    applications: int = 0


class RoleUpdate(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}")
        return v


class ModelCard(BaseModel):
    source: str
    loaded: bool
    model_name: Optional[str] = None
    trained_at: Optional[str] = None
    n_features: int = 0
    features: List[str] = []
    network_features: List[str] = []
    bands: dict = {}
    flag_thresholds: dict = {}
    metrics: dict = {}
    n_members: int = 0
    n_borrowers_with_loans: int = 0


class ActivityStats(BaseModel):
    users_total: int
    users_by_role: dict = {}
    applications_total: int
    applications_by_status: dict = {}
    applications_by_band: dict = {}


class InsightsOverview(BaseModel):
    # portfolio
    n_loans: int
    n_members: int
    total_disbursed: float
    outcomes: dict = {}          # {"Repaid": n, "Written off": n, ...}
    branches: dict = {}          # {"Gasabo": n, ...}
    bad_rate: float              # share of matured loans that went bad (0-1)
    written_off_value: float
    n_arrears: int
    arrears_value: float
    # guarantor network
    unique_guarantors: int
    avg_guarantors: float
    over_committed: int
    ever_defaulted: int
    loans_backed_by_defaulter: int
    pct_backed_by_defaulter: float
    n_communities: int
    worst_community_rate: float
