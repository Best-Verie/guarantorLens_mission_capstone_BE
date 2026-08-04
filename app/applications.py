"""Loan-application workflow: create + assess, list, escalate, recommend.

An application saves an assessment so it can be reviewed, escalated from a branch
officer to head-office credit staff, and have recommendations attached. The system
never makes the lending decision, it only records recommendations (decision support).
"""
import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from . import scoring
from .auth import get_current_user
from .db import get_db
from .models import Application, AuditLog, Recommendation, User
from .schema import (
    ApplicationCreate, ApplicationListItem, ApplicationOut, ApplicationStats,
    AuditLogOut, EscalateRequest, RecommendationCreate, RecommendationOut,
)

router = APIRouter(tags=["applications"])

MANAGER_ROLES = {"credit_manager", "admin"}


def _record_audit(db: Session, user: User, action: str, app: Application, detail: dict) -> None:
    """Append one immutable audit-log row for a decision or override. Never updates."""
    db.add(AuditLog(application_id=app.id, actor_id=user.id, actor_name=user.full_name,
                    actor_role=user.role, action=action, detail=json.dumps(detail)))


def _is_manager(user: User) -> bool:
    return user.role in MANAGER_ROLES


def _iso(dt):
    return dt.isoformat() if dt is not None else None


def _rec_out(r: Recommendation) -> dict:
    return {"id": r.id, "author_name": r.author_name, "author_role": r.author_role,
            "decision": r.decision, "note": r.note, "created_at": _iso(r.created_at)}


def _app_out(a: Application) -> dict:
    return {
        "id": a.id, "created_by_name": a.created_by_name, "branch": a.branch,
        "borrower_id": a.borrower_id, "borrower_name": a.borrower_name,
        "amount": a.amount, "savings": a.savings, "salary": a.salary,
        "interest_rate": a.interest_rate,
        "guarantor_ids": json.loads(a.guarantor_ids) if a.guarantor_ids else [],
        "guarantor_overrides": json.loads(a.guarantor_overrides) if a.guarantor_overrides else None,
        "risk_score": a.risk_score, "band": a.band, "probability": a.probability,
        "reasons": json.loads(a.reasons) if a.reasons else [],
        "flags": json.loads(a.flags) if a.flags else [],
        "segment": json.loads(a.segment) if a.segment else None,
        "unusual": json.loads(a.unusual) if a.unusual else None,
        "source": a.source, "status": a.status, "escalation_note": a.escalation_note,
        "created_at": _iso(a.created_at),
        "recommendations": [_rec_out(r) for r in a.recommendations],
    }


@router.post("/applications", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED)
def create_application(body: ApplicationCreate, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    if not body.amount or body.amount <= 0:
        raise HTTPException(status_code=400, detail="Enter a loan amount.")
    result = scoring.assess(body.amount, body.savings, body.salary, None,
                            body.guarantor_ids, borrower_id=body.borrower_id,
                            interest_rate=body.interest_rate,
                            guarantor_overrides=body.guarantor_overrides)
    branch = body.branch
    if not branch and body.borrower_id and "-" in body.borrower_id:
        branch = body.borrower_id.split("-")[0]
    app_row = Application(
        created_by=user.id, created_by_name=user.full_name, branch=branch,
        borrower_id=body.borrower_id, borrower_name=body.borrower_name,
        amount=body.amount, savings=body.savings, salary=body.salary,
        interest_rate=body.interest_rate,
        guarantor_ids=json.dumps(body.guarantor_ids or []),
        guarantor_overrides=json.dumps(body.guarantor_overrides) if body.guarantor_overrides else None,
        risk_score=result["risk_score"], band=result["band"], probability=result["probability"],
        reasons=json.dumps(result["reasons"]), flags=json.dumps(result["flags"]),
        segment=json.dumps(result.get("segment")) if result.get("segment") else None,
        unusual=json.dumps(result.get("unusual")) if result.get("unusual") else None,
        source=result["source"], status="assessed",
    )
    db.add(app_row); db.commit(); db.refresh(app_row)
    _record_audit(db, user, "assess", app_row, {
        "band": app_row.band, "risk_score": app_row.risk_score, "source": app_row.source,
        "amount": app_row.amount,
        "guarantor_overrides": body.guarantor_overrides or None,   # what-if overrides applied at assess time
    })
    db.commit()
    return _app_out(app_row)


@router.get("/applications", response_model=list[ApplicationListItem])
def list_applications(status: str | None = None, escalated: bool = False,
                      db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = db.query(Application)
    if not _is_manager(user):
        q = q.filter(Application.created_by == user.id)   # officers see their own
    if escalated:
        q = q.filter(Application.status == "escalated")
    elif status:
        q = q.filter(Application.status == status)
    rows = q.order_by(Application.created_at.desc()).limit(500).all()
    return [{"id": a.id, "borrower": a.borrower_name or a.borrower_id, "branch": a.branch,
             "amount": a.amount, "risk_score": a.risk_score, "band": a.band,
             "status": a.status, "created_by_name": a.created_by_name,
             "created_at": _iso(a.created_at)} for a in rows]


@router.get("/applications/stats", response_model=ApplicationStats)
def application_stats(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    escalated = db.query(Application).filter(Application.status == "escalated").count()
    my_open = db.query(Application).filter(
        Application.created_by == user.id,
        Application.status.in_(["assessed", "escalated"])).count()
    total = db.query(Application).count() if _is_manager(user) else \
        db.query(Application).filter(Application.created_by == user.id).count()
    return {"my_open": my_open, "escalated": escalated, "total": total}


@router.get("/applications/{app_id}", response_model=ApplicationOut)
def get_application(app_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    a = db.get(Application, app_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    if not _is_manager(user) and a.created_by != user.id:
        raise HTTPException(status_code=403, detail="Not allowed to view this application.")
    return _app_out(a)


@router.post("/applications/{app_id}/escalate", response_model=ApplicationOut)
def escalate_application(app_id: int, body: EscalateRequest, db: Session = Depends(get_db),
                         user: User = Depends(get_current_user)):
    a = db.get(Application, app_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    if not _is_manager(user) and a.created_by != user.id:
        raise HTTPException(status_code=403, detail="Not allowed.")
    a.status = "escalated"
    a.escalation_note = (body.note or "").strip() or None
    _record_audit(db, user, "escalate", a, {"note": a.escalation_note})
    db.commit(); db.refresh(a)
    return _app_out(a)


@router.post("/applications/{app_id}/recommendations", response_model=ApplicationOut)
def add_recommendation(app_id: int, body: RecommendationCreate, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    a = db.get(Application, app_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    # Separation of duties: only a credit manager (or admin) records the recommendation.
    # A loan officer proposes and escalates; they cannot approve/decline (least of all their own).
    if not _is_manager(user):
        raise HTTPException(status_code=403,
                            detail="Only a credit manager can add a recommendation.")
    if body.decision not in {"approve", "request_changes", "decline"}:
        raise HTTPException(status_code=400, detail="Invalid recommendation.")
    rec = Recommendation(application_id=a.id, author_id=user.id, author_name=user.full_name,
                         author_role=user.role, decision=body.decision,
                         note=(body.note or "").strip() or None)
    db.add(rec)
    a.status = "recommended"
    _record_audit(db, user, "recommend", a, {"decision": body.decision,
                                             "note": (body.note or "").strip() or None})
    db.commit(); db.refresh(a)
    return _app_out(a)


@router.get("/applications/{app_id}/audit", response_model=list[AuditLogOut])
def get_audit_log(app_id: int, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)):
    """Read the append-only audit trail for an application (credit manager / admin only)."""
    a = db.get(Application, app_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    if not _is_manager(user) and a.created_by != user.id:
        raise HTTPException(status_code=403, detail="Not allowed to view this audit log.")
    rows = (db.query(AuditLog).filter(AuditLog.application_id == app_id)
            .order_by(AuditLog.created_at, AuditLog.id).all())
    return [{"id": r.id, "application_id": r.application_id, "actor_name": r.actor_name,
             "actor_role": r.actor_role, "action": r.action,
             "detail": json.loads(r.detail) if r.detail else None,
             "created_at": _iso(r.created_at)} for r in rows]
