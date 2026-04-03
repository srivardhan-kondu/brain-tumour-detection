"""
Authentication module – JWT-based login/signup for Brain Tumor Segmentation App.
Users are stored in a local JSON file (users.json) for simplicity.
"""

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

USERS_FILE = Path(__file__).parent / "users.json"

# ─────────────────────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field(default="Radiologist", max_length=50)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserOut(BaseModel):
    name: str
    email: str
    role: str
    initials: str


# ─────────────────────────────────────────────────────────────
# User storage (JSON file)
# ─────────────────────────────────────────────────────────────

def _load_users() -> list[dict]:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return []


def _save_users(users: list[dict]):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _find_user(email: str) -> Optional[dict]:
    for u in _load_users():
        if u["email"].lower() == email.lower():
            return u
    return None


def _make_initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


# ─────────────────────────────────────────────────────────────
# Token operations
# ─────────────────────────────────────────────────────────────

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────
# Auth dependency (optional – does not block if no token)
# ─────────────────────────────────────────────────────────────

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    """Returns user dict if valid token, else None. Does NOT raise."""
    if not token:
        return None
    payload = decode_token(token)
    if not payload or "email" not in payload:
        return None
    user = _find_user(payload["email"])
    if not user:
        return None
    return {"name": user["name"], "email": user["email"], "role": user["role"], "initials": _make_initials(user["name"])}


async def require_auth(token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """Raises 401 if no valid token."""
    user = await get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ─────────────────────────────────────────────────────────────
# Signup / Login
# ─────────────────────────────────────────────────────────────

def signup_user(req: SignupRequest) -> TokenResponse:
    if _find_user(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(req.password)
    user = {
        "name": req.name,
        "email": req.email,
        "role": req.role,
        "password_hash": hashed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    users = _load_users()
    users.append(user)
    _save_users(users)

    token = create_access_token({"email": user["email"], "name": user["name"]})
    return TokenResponse(
        access_token=token,
        user={"name": user["name"], "email": user["email"], "role": user["role"], "initials": _make_initials(user["name"])},
    )


def login_user(req: LoginRequest) -> TokenResponse:
    user = _find_user(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"email": user["email"], "name": user["name"]})
    return TokenResponse(
        access_token=token,
        user={"name": user["name"], "email": user["email"], "role": user["role"], "initials": _make_initials(user["name"])},
    )
