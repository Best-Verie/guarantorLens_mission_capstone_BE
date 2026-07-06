"""Admin-only endpoints (role == 'admin'): user governance, the deployed-model
card, activity counts, and hot-swapping the served model artifacts.

The model itself is trained offline in the notebook. The admin's job here is the
*lifecycle*: see what is deployed, and deploy a new bundle produced by retraining.
"""
import json
import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from . import network_data, scoring
from .auth import get_current_user
from .db import get_db
from .models import Application, Recommendation, User
from .schema import (
    ActivityStats, AdminUserOut, MessageResponse, ModelCard, RoleUpdate,
)

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only.")
    return user


def _iso(dt):
    return dt.isoformat() if dt is not None else None


# --- users ------------------------------------------------------------------

@router.get("/users", response_model=list[AdminUserOut])
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.asc()).all()
    author_counts: dict = {}
    for (uid,) in db.query(Application.created_by).all():
        author_counts[uid] = author_counts.get(uid, 0) + 1
    return [
        AdminUserOut(
            id=u.id, full_name=u.full_name, email=u.email, role=u.role,
            created_at=_iso(u.created_at), applications=author_counts.get(u.id, 0),
        )
        for u in users
    ]


@router.patch("/users/{user_id}/role", response_model=AdminUserOut)
def set_role(user_id: int, body: RoleUpdate,
             admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if target.role == "admin" and body.role != "admin" and _admin_count(db) <= 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "This is the only admin. Promote another admin first.")
    target.role = body.role
    db.commit()
    db.refresh(target)
    return AdminUserOut(id=target.id, full_name=target.full_name, email=target.email,
                        role=target.role, created_at=_iso(target.created_at))


@router.delete("/users/{user_id}", response_model=MessageResponse)
def delete_user(user_id: int,
                admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot delete your own account.")
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if target.role == "admin" and _admin_count(db) <= 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot delete the only admin.")
    # Keep their work: reassign authored rows to the acting admin (the denormalised
    # created_by_name / author_name preserve who originally did it).
    db.query(Application).filter(Application.created_by == user_id).update(
        {Application.created_by: admin.id}, synchronize_session=False)
    db.query(Recommendation).filter(Recommendation.author_id == user_id).update(
        {Recommendation.author_id: admin.id}, synchronize_session=False)
    db.delete(target)
    db.commit()
    return MessageResponse(message=f"Removed {target.full_name}. Their assessments were reassigned to you.")


def _admin_count(db: Session) -> int:
    return db.query(User).filter(User.role == "admin").count()


# --- deployed model ---------------------------------------------------------

@router.get("/model", response_model=ModelCard)
def model_card(admin: User = Depends(require_admin)):
    return scoring.model_info()


@router.post("/model", response_model=ModelCard)
async def update_model(
    model: UploadFile = File(..., description="joblib bundle with 'model' and 'features'"),
    members: Optional[UploadFile] = File(None, description="members JSON (optional)"),
    loans: Optional[UploadFile] = File(None, description="loans JSON (optional)"),
    admin: User = Depends(require_admin),
):
    """Validate an uploaded bundle, then swap it in and reload the scorer. Rejects the
    upload before touching the live files if the bundle is not a valid GuarantorLens model."""
    tmpdir = tempfile.mkdtemp()
    try:
        mpath = os.path.join(tmpdir, "model.joblib")
        with open(mpath, "wb") as fh:
            fh.write(await model.read())
        try:
            import joblib
            bundle = joblib.load(mpath)
        except Exception:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Could not read the model file (expected a joblib bundle).")
        if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Bundle must be a dict containing 'model' and 'features'.")

        members_path = _validated_json(members, tmpdir, "members.json") if members else None
        loans_path = _validated_json(loans, tmpdir, "loans.json") if loans else None

        # everything validated -> move into place
        shutil.move(mpath, scoring.MODEL_PATH)
        if members_path:
            shutil.move(members_path, scoring.MEMBERS_PATH)
        if loans_path:
            shutil.move(loans_path, scoring.LOANS_PATH)

        scoring.reload()
        if loans_path:
            network_data.reload()
        info = scoring.model_info()
        if not info["loaded"]:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "The new bundle failed to load after swapping.")
        return info
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _validated_json(upload: UploadFile, tmpdir: str, name: str) -> str:
    raw = upload.file.read()
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{name} is not valid JSON.")
    if not isinstance(data, list):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{name} must be a JSON list of records.")
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as fh:
        fh.write(raw)
    return path


# --- activity ---------------------------------------------------------------

@router.get("/activity", response_model=ActivityStats)
def activity(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users_by_role: dict = {}
    for (role,) in db.query(User.role).all():
        users_by_role[role] = users_by_role.get(role, 0) + 1
    by_status: dict = {}
    by_band: dict = {}
    for (st, band) in db.query(Application.status, Application.band).all():
        by_status[st] = by_status.get(st, 0) + 1
        if band:
            by_band[band] = by_band.get(band, 0) + 1
    return ActivityStats(
        users_total=sum(users_by_role.values()),
        users_by_role=users_by_role,
        applications_total=sum(by_status.values()),
        applications_by_status=by_status,
        applications_by_band=by_band,
    )
