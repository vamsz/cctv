"""FastAPI dependencies: auth, RBAC, rate limit."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select

from config.settings import settings
from src.common.db import session_scope
from src.common.security import JWTError, verify_jwt
from src.evidence.models import User


def get_bearer(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return authorization.split(" ", 1)[1]


def current_user(token: Annotated[str, Depends(get_bearer)]) -> User:
    try:
        claims = verify_jwt(token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc
    with session_scope() as s:
        user = s.scalar(select(User).where(User.email == claims.get("sub")))
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="user disabled")
        # detach so the request can read fields after session closes
        s.expunge(user)
        return user


def require_role(*roles: str):
    def _dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles and "admin" not in {user.role}:
            raise HTTPException(status_code=403, detail="forbidden")
        return user
    return _dep


# ---------------- rate limiting (per-IP, sliding-window, in-process) ---------

_window = 60.0
_buckets: dict[str, deque] = defaultdict(deque)
_lock = Lock()


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "?"
    now = time.time()
    with _lock:
        q = _buckets[ip]
        while q and now - q[0] > _window:
            q.popleft()
        if len(q) >= settings.rate_limit_per_minute:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")
        q.append(now)
