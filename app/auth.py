"""Authentication endpoints: register, login, password reset.

All routes are under /auth. Access tokens are JWTs sent as a Bearer header.
The password-reset flow stores only a hash of the emailed token and always
answers /forgot-password with 200 so it cannot be used to discover which
emails have accounts.
"""
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from . import security
from .db import get_db
from .models import User
from .schema import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=True)

DEBUG = os.getenv("DEBUG", "False").strip().lower() in {"1", "true", "yes"}
RESET_TOKEN_TTL_MIN = int(os.getenv("RESET_TOKEN_TTL_MIN", "60"))


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the Bearer token to a user, or 401."""
    try:
        payload = security.decode_access_token(creds.credentials)
        user_id = int(payload["sub"])
    except Exception:  # invalid signature, expired, malformed sub
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
        )
    return user


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    user = User(
        full_name=body.full_name.strip(),
        email=body.email,
        role=body.role,
        hashed_password=security.hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = security.create_access_token(str(user.id))
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not security.verify_password(body.password, user.hashed_password):
        # Same message either way, so we don't reveal whether the email exists.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    token = security.create_access_token(str(user.id))
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.model_validate(user)


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    reset_token = None
    if user is not None:
        reset_token = security.generate_reset_token()
        user.reset_token_hash = security.hash_reset_token(reset_token)
        user.reset_token_expires = datetime.now(timezone.utc) + timedelta(
            minutes=RESET_TOKEN_TTL_MIN
        )
        db.commit()
        # TODO: email the link, e.g. https://<frontend>/reset-password?token=<reset_token>
    return ForgotPasswordResponse(
        message="If that email matches an account, a reset link has been sent.",
        reset_token=reset_token if DEBUG else None,
    )


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash = security.hash_reset_token(body.token)
    user = db.query(User).filter(User.reset_token_hash == token_hash).first()
    if user is None or user.reset_token_expires is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This reset link is invalid.")

    expires = user.reset_token_expires
    if expires.tzinfo is None:  # SQLite returns naive datetimes
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This reset link has expired.")

    user.hashed_password = security.hash_password(body.new_password)
    user.reset_token_hash = None
    user.reset_token_expires = None
    db.commit()
    return MessageResponse(message="Your password has been updated. You can now sign in.")
