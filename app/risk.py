"""Loan risk assessment endpoint."""
from fastapi import APIRouter, Depends

from . import scoring
from .auth import get_current_user
from .models import User
from .schema import AssessRequest, AssessResponse

router = APIRouter(tags=["risk"])


@router.post("/assess-risk", response_model=AssessResponse)
def assess_risk(body: AssessRequest, user: User = Depends(get_current_user)):
    """Score a loan from the borrower details and their guarantor network.

    Returns a risk score, band, plain-language reasons, and guarantor-network
    flags. This is decision support for the officer, not an automatic approval.
    """
    return scoring.assess(
        amount=body.amount,
        savings=body.savings,
        salary=body.salary,
        disb_date=body.disbursement_date,
        guarantor_ids=body.guarantor_ids,
        borrower_id=body.borrower_id,
        interest_rate=body.interest_rate,
        guarantor_overrides=body.guarantor_overrides,
    )


@router.post("/assess/suggest-guarantors")
def suggest_guarantors(body: AssessRequest, user: User = Depends(get_current_user)):
    """For a Medium/High loan, suggest a single guarantor change (swap the weakest backer, or add
    one) that would lower the risk band, with the re-scored result. Advice, not a decision."""
    return scoring.suggest_guarantors(
        amount=body.amount,
        savings=body.savings,
        salary=body.salary,
        guarantor_ids=body.guarantor_ids,
        borrower_id=body.borrower_id,
        interest_rate=body.interest_rate,
    )
