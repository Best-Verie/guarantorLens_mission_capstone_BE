"""Password hashing, JWT access tokens, and password-reset tokens.

Password hashing uses the standard library (PBKDF2-HMAC-SHA256) so there is no
bcrypt/argon2 build step to fail on Render. Access tokens are signed JWTs.
"""
import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_TTL_MIN = int(os.getenv("ACCESS_TOKEN_TTL_MIN", "720"))  # 12 hours
_PBKDF2_ITERATIONS = 240_000


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# --- passwords --------------------------------------------------------------

def hash_password(password: str) -> str:
    """Return a self-describing hash: pbkdf2_sha256$iterations$salt$hash."""
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${_b64encode(salt)}${_b64encode(derived)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _b64decode(salt_b64), int(iterations)
        )
        return hmac.compare_digest(derived, _b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


# --- access tokens (JWT) ----------------------------------------------------

def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_TTL_MIN),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALG)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALG])


# --- password-reset tokens --------------------------------------------------
# We email the raw token to the user but store only its SHA-256, so a leaked
# database cannot be used to reset passwords.

def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
