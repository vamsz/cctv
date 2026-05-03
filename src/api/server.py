"""Hardened FastAPI app.

Security posture:
  - JWT bearer auth, RBAC roles: admin | operator | auditor
  - bcrypt passwords; no plaintext anywhere
  - per-IP rate limiting
  - CORS locked to ALLOWED_ORIGINS
  - audit log on every state change
  - healthz / readyz / metrics
  - structured request logs with request_id
  - error responses follow a single Problem schema
"""
from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
import structlog

from config.settings import settings
from src.common.audit import record as audit
from src.common.db import session_scope
from src.common.logging import configure_logging, get_logger
from src.common.metrics import (
    api_request_latency,
    review_action_total,
    review_backlog,
    start_metrics_server,
)
from src.common.object_store import get_object_store
from src.common.security import hash_password, issue_jwt, verify_password
from src.common.signing import verify as verify_signature
from src.common.tracing import init_sentry
from src.evidence.models import CameraHealth, User, Violation
from src.evidence.store import ReviewStatus
from .deps import current_user, rate_limit, require_role

# ----------------------------------------------------------------- app setup

configure_logging()
init_sentry()
log = get_logger("api")

app = FastAPI(title="CCTV Enforcement Review", version="1.0.0", docs_url="/api/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["authorization", "content-type"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    structlog.contextvars.bind_contextvars(request_id=rid, path=request.url.path, method=request.method)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dur = time.perf_counter() - t0
        status_code = locals().get("response").status_code if locals().get("response") else 500
        api_request_latency.labels(request.method, request.url.path, str(status_code)).observe(dur)
        log.info("http_request", duration_ms=int(dur * 1000), status=status_code)
        structlog.contextvars.unbind_contextvars("request_id", "path", "method")


# ----------------------------------------------------------------- error model


class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: Optional[str] = None
    request_id: Optional[str] = None


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=Problem(title=str(exc.detail), status=exc.status_code,
                        request_id=request.headers.get("x-request-id")).model_dump(),
    )


# ----------------------------------------------------------------- schemas


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class TokenOut(BaseModel):
    token: str
    role: str
    expires_in: int


class ReviewIn(BaseModel):
    notes: Optional[str] = Field(default=None, max_length=1024)


class BulkReviewIn(BaseModel):
    ids: list[int]
    notes: Optional[str] = Field(default=None, max_length=1024)


class ViolationOut(BaseModel):
    id: int
    code: str
    camera_id: str
    track_id: int
    timestamp: datetime
    plate_text: Optional[str]
    plate_ocr_confidence: Optional[float]
    rule_confidence: float
    frame_url: str
    annotated_url: Optional[str]
    plate_crop_url: Optional[str]
    status: str
    reviewer_id: Optional[int]
    reviewed_at: Optional[datetime]
    review_notes: Optional[str]
    sha256: str
    chain_hash: str
    pushed_to_chalana: bool


class CameraOut(BaseModel):
    camera_id: str
    is_up: bool
    last_frame_at: Optional[datetime]
    last_violation_at: Optional[datetime]
    fps_observed: Optional[float]
    last_error: Optional[str]


class StatsOut(BaseModel):
    total: int
    pending: int
    approved: int
    rejected: int
    by_code: dict[str, int]
    by_camera: dict[str, int]
    last_24h: int


# ----------------------------------------------------------------- helpers


def _to_out(v: Violation) -> ViolationOut:
    def url(p: Optional[str]) -> Optional[str]:
        return f"/api/evidence/{p}" if p else None
    return ViolationOut(
        id=v.id,
        code=v.code,
        camera_id=v.camera_id,
        track_id=v.track_id,
        timestamp=v.timestamp,
        plate_text=v.plate_text,
        plate_ocr_confidence=v.plate_ocr_confidence,
        rule_confidence=v.rule_confidence,
        frame_url=url(v.frame_path) or "",
        annotated_url=url(v.annotated_path),
        plate_crop_url=url(v.plate_crop_path),
        status=v.status,
        reviewer_id=v.reviewer_id,
        reviewed_at=v.reviewed_at,
        review_notes=v.review_notes,
        sha256=v.sha256,
        chain_hash=v.chain_hash,
        pushed_to_chalana=v.pushed_to_chalana,
    )


# ----------------------------------------------------------------- auth


@app.post("/api/auth/login", response_model=TokenOut, dependencies=[Depends(rate_limit)])
def login(body: LoginIn, request: Request):
    with session_scope() as s:
        user = s.scalar(select(User).where(User.email == body.email))
        if not user or not user.is_active or not verify_password(body.password, user.password_hash):
            audit("auth.login_failed", actor_label=body.email, ip_address=request.client.host if request.client else None)
            raise HTTPException(status_code=401, detail="invalid credentials")
        user.last_login_at = datetime.utcnow()
        token = issue_jwt(user.email, user.role)
        audit("auth.login", actor_id=user.id, actor_label=user.email,
              ip_address=request.client.host if request.client else None)
        return TokenOut(token=token, role=user.role, expires_in=settings.jwt_ttl_minutes * 60)


# ----------------------------------------------------------------- violations


@app.get("/api/violations", response_model=list[ViolationOut], dependencies=[Depends(rate_limit)])
def list_violations(
    user: User = Depends(current_user),
    status: Optional[str] = Query(default=None, pattern="^(pending|approved|rejected)$"),
    code: Optional[str] = None,
    camera_id: Optional[str] = None,
    plate: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
):
    with session_scope() as s:
        stmt = select(Violation).order_by(Violation.timestamp.desc())
        if status: stmt = stmt.where(Violation.status == status)
        if code: stmt = stmt.where(Violation.code == code)
        if camera_id: stmt = stmt.where(Violation.camera_id == camera_id)
        if plate: stmt = stmt.where(Violation.plate_text.like(f"%{plate.upper()}%"))
        if since: stmt = stmt.where(Violation.timestamp >= since)
        rows = s.scalars(stmt.offset(offset).limit(limit)).all()
        return [_to_out(r) for r in rows]


@app.get("/api/violations/{vid}", response_model=ViolationOut)
def get_violation(vid: int, user: User = Depends(current_user)):
    with session_scope() as s:
        v = s.get(Violation, vid)
        if not v:
            raise HTTPException(status_code=404, detail="not found")
        return _to_out(v)


@app.post("/api/violations/{vid}/approve", response_model=ViolationOut)
def approve(vid: int, body: ReviewIn, request: Request, user: User = Depends(require_role("operator"))):
    return _set_status(vid, ReviewStatus.APPROVED, body, user, request)


@app.post("/api/violations/{vid}/reject", response_model=ViolationOut)
def reject(vid: int, body: ReviewIn, request: Request, user: User = Depends(require_role("operator"))):
    return _set_status(vid, ReviewStatus.REJECTED, body, user, request)


@app.post("/api/violations/bulk_approve")
def bulk_approve(body: BulkReviewIn, request: Request, user: User = Depends(require_role("operator"))):
    return _bulk(body, ReviewStatus.APPROVED, user, request)


@app.post("/api/violations/bulk_reject")
def bulk_reject(body: BulkReviewIn, request: Request, user: User = Depends(require_role("operator"))):
    return _bulk(body, ReviewStatus.REJECTED, user, request)


def _set_status(vid: int, target: ReviewStatus, body: ReviewIn, user: User, request: Request) -> ViolationOut:
    with session_scope() as s:
        v = s.get(Violation, vid)
        if not v:
            raise HTTPException(status_code=404, detail="not found")
        if v.status != "pending":
            raise HTTPException(status_code=409, detail=f"already {v.status}")
        v.status = target.value
        v.reviewer_id = user.id
        v.reviewed_at = datetime.utcnow()
        v.review_notes = body.notes
        s.flush()
        review_action_total.labels(action=target.value).inc()
        audit(f"violation.{target.value}",
              actor_id=user.id, actor_label=user.email,
              target_type="violation", target_id=str(v.id),
              detail={"notes": body.notes},
              ip_address=request.client.host if request.client else None)
        return _to_out(v)


def _bulk(body: BulkReviewIn, target: ReviewStatus, user: User, request: Request) -> dict:
    if not body.ids:
        return {"updated": 0}
    with session_scope() as s:
        rows = s.scalars(select(Violation).where(Violation.id.in_(body.ids))).all()
        updated = 0
        for v in rows:
            if v.status != "pending":
                continue
            v.status = target.value
            v.reviewer_id = user.id
            v.reviewed_at = datetime.utcnow()
            v.review_notes = body.notes
            review_action_total.labels(action=target.value).inc()
            updated += 1
        audit(f"violation.bulk_{target.value}",
              actor_id=user.id, actor_label=user.email,
              detail={"ids": body.ids, "updated": updated},
              ip_address=request.client.host if request.client else None)
    return {"updated": updated}


@app.get("/api/violations/{vid}/verify")
def verify_violation(vid: int, user: User = Depends(current_user)):
    """Re-verifies the Ed25519 signature against the row payload."""
    with session_scope() as s:
        v = s.get(Violation, vid)
        if not v:
            raise HTTPException(status_code=404, detail="not found")
        ok = verify_signature(
            {
                "code": v.code,
                "camera_id": v.camera_id,
                "track_id": v.track_id,
                "timestamp": v.timestamp.timestamp(),
                "frame_idx": v.frame_idx,
                "plate_text": v.plate_text,
                "rule_confidence": v.rule_confidence,
                "deployment_id": settings.deployment_id,
            },
            v.sha256,
            bytes(v.signature),
        )
        return {"id": v.id, "signature_valid": ok, "sha256": v.sha256, "chain_hash": v.chain_hash}


# ----------------------------------------------------------------- evidence streaming


@app.get("/api/evidence/{key:path}")
def evidence(key: str, user: User = Depends(current_user)):
    store = get_object_store()
    if not store.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    data = store.get_bytes(key)
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


# ----------------------------------------------------------------- stats / cameras


@app.get("/api/stats", response_model=StatsOut)
def stats(user: User = Depends(current_user)):
    with session_scope() as s:
        total = s.scalar(select(func.count(Violation.id))) or 0
        pending = s.scalar(select(func.count(Violation.id)).where(Violation.status == "pending")) or 0
        approved = s.scalar(select(func.count(Violation.id)).where(Violation.status == "approved")) or 0
        rejected = s.scalar(select(func.count(Violation.id)).where(Violation.status == "rejected")) or 0
        last_24h = s.scalar(
            select(func.count(Violation.id)).where(Violation.timestamp >= datetime.utcnow() - timedelta(hours=24))
        ) or 0
        by_code = dict(s.execute(select(Violation.code, func.count(Violation.id)).group_by(Violation.code)).all())
        by_camera = dict(s.execute(select(Violation.camera_id, func.count(Violation.id)).group_by(Violation.camera_id)).all())
        review_backlog.set(pending)
        return StatsOut(total=total, pending=pending, approved=approved, rejected=rejected,
                        by_code=by_code, by_camera=by_camera, last_24h=last_24h)


@app.get("/api/cameras", response_model=list[CameraOut])
def cameras(user: User = Depends(current_user)):
    with session_scope() as s:
        rows = s.scalars(select(CameraHealth)).all()
        return [
            CameraOut(camera_id=r.camera_id, is_up=r.is_up, last_frame_at=r.last_frame_at,
                      last_violation_at=r.last_violation_at, fps_observed=r.fps_observed,
                      last_error=r.last_error)
            for r in rows
        ]


# ----------------------------------------------------------------- health / metrics


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": app.version, "deployment": settings.deployment_id}


@app.get("/readyz")
def readyz():
    # DB reachable?
    try:
        with session_scope() as s:
            s.execute(select(func.count(User.id)))
    except Exception as exc:
        return JSONResponse(status_code=503, content={"ok": False, "error": str(exc)})
    return {"ok": True}


# ----------------------------------------------------------------- UI


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return (STATIC_DIR / "login.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------- startup


@app.on_event("startup")
def on_startup():
    settings.assert_production_ready()
    if settings.prometheus_port:
        try:
            start_metrics_server(settings.prometheus_port)
        except OSError:
            log.warning("prometheus_port_in_use", port=settings.prometheus_port)
    _ensure_bootstrap_admin()


def _ensure_bootstrap_admin():
    with session_scope() as s:
        n = s.scalar(select(func.count(User.id))) or 0
        if n > 0:
            return
        s.add(User(
            email=settings.bootstrap_admin_email,
            password_hash=hash_password(settings.bootstrap_admin_password.get_secret_value()),
            role="admin",
            full_name="Bootstrap admin",
        ))
        log.warning("bootstrap_admin_created", email=settings.bootstrap_admin_email)
