"""Auth primitives: bcrypt password hashing + JWT issue/verify."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from config.settings import settings

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except ValueError:
        return False


def issue_jwt(subject: str, role: str, extra: dict[str, Any] | None = None) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_ttl_minutes)).timestamp()),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        **(extra or {}),
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm="HS256")


def verify_jwt(token: str) -> dict[str, Any]:
    return jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
    )


JWTError = JWTError  # re-export for callers
