"""FastAPI review console — no auth, focus on working product.

Endpoints:
  GET  /api/violations         list with filters
  GET  /api/violations/{id}    single violation
  POST /api/violations/{id}/approve
  POST /api/violations/{id}/reject
  POST /api/violations/bulk_approve
  POST /api/violations/bulk_reject
  GET  /api/evidence/{key}     serve evidence images
  GET  /api/stats              dashboard stats
  GET  /api/cameras            camera health
  GET  /healthz                liveness probe
  GET  /                       review dashboard UI
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config.settings import settings
from src.common.db import session_scope
from src.common.logging import configure_logging, get_logger
from src.common.metrics import review_action_total, review_backlog, start_metrics_server
from src.common.object_store import get_object_store
from src.evidence.models import Base, CameraHealth, Violation
from src.evidence.store import ReviewStatus

configure_logging()
log = get_logger("api")

app = FastAPI(title="CCTV Enforcement Review", version="1.0.0", docs_url="/api/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------------------------------------------------- schemas


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
    reviewed_at: Optional[datetime]
    review_notes: Optional[str]
    sha256: str
    chain_hash: str


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
        reviewed_at=v.reviewed_at,
        review_notes=v.review_notes,
        sha256=v.sha256,
        chain_hash=v.chain_hash,
    )


# ----------------------------------------------------------------- violations


@app.get("/api/violations", response_model=list[ViolationOut])
def list_violations(
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
        if status:
            stmt = stmt.where(Violation.status == status)
        if code:
            stmt = stmt.where(Violation.code == code)
        if camera_id:
            stmt = stmt.where(Violation.camera_id == camera_id)
        if plate:
            stmt = stmt.where(Violation.plate_text.like(f"%{plate.upper()}%"))
        if since:
            stmt = stmt.where(Violation.timestamp >= since)
        rows = s.scalars(stmt.offset(offset).limit(limit)).all()
        return [_to_out(r) for r in rows]


@app.get("/api/violations/{vid}", response_model=ViolationOut)
def get_violation(vid: int):
    with session_scope() as s:
        v = s.get(Violation, vid)
        if not v:
            raise HTTPException(status_code=404, detail="not found")
        return _to_out(v)


@app.post("/api/violations/{vid}/approve", response_model=ViolationOut)
def approve(vid: int, body: ReviewIn):
    return _set_status(vid, ReviewStatus.APPROVED, body)


@app.post("/api/violations/{vid}/reject", response_model=ViolationOut)
def reject(vid: int, body: ReviewIn):
    return _set_status(vid, ReviewStatus.REJECTED, body)


@app.post("/api/violations/bulk_approve")
def bulk_approve(body: BulkReviewIn):
    return _bulk(body, ReviewStatus.APPROVED)


@app.post("/api/violations/bulk_reject")
def bulk_reject(body: BulkReviewIn):
    return _bulk(body, ReviewStatus.REJECTED)


def _set_status(vid: int, target: ReviewStatus, body: ReviewIn) -> ViolationOut:
    with session_scope() as s:
        v = s.get(Violation, vid)
        if not v:
            raise HTTPException(status_code=404, detail="not found")
        if v.status != "pending":
            raise HTTPException(status_code=409, detail=f"already {v.status}")
        v.status = target.value
        v.reviewed_at = datetime.utcnow()
        v.review_notes = body.notes
        s.flush()
        review_action_total.labels(action=target.value).inc()
        return _to_out(v)


def _bulk(body: BulkReviewIn, target: ReviewStatus) -> dict:
    if not body.ids:
        return {"updated": 0}
    with session_scope() as s:
        rows = s.scalars(select(Violation).where(Violation.id.in_(body.ids))).all()
        updated = 0
        for v in rows:
            if v.status != "pending":
                continue
            v.status = target.value
            v.reviewed_at = datetime.utcnow()
            v.review_notes = body.notes
            review_action_total.labels(action=target.value).inc()
            updated += 1
    return {"updated": updated}


# ----------------------------------------------------------------- evidence streaming


@app.get("/api/evidence/{key:path}")
def evidence(key: str):
    store = get_object_store()
    if not store.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    data = store.get_bytes(key)
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


# ----------------------------------------------------------------- stats / cameras


@app.get("/api/stats", response_model=StatsOut)
def stats():
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
def cameras():
    with session_scope() as s:
        rows = s.scalars(select(CameraHealth)).all()
        return [
            CameraOut(camera_id=r.camera_id, is_up=r.is_up, last_frame_at=r.last_frame_at,
                      last_violation_at=r.last_violation_at, fps_observed=r.fps_observed,
                      last_error=r.last_error)
            for r in rows
        ]


# ----------------------------------------------------------------- health


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": app.version, "deployment": settings.deployment_id}


# ----------------------------------------------------------------- UI


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------- startup


@app.on_event("startup")
def on_startup():
    from src.common.db import engine
    Base.metadata.create_all(bind=engine())
    log.info("db_tables_created")
    if settings.prometheus_port:
        try:
            start_metrics_server(settings.prometheus_port)
        except OSError:
            log.warning("prometheus_port_in_use", port=settings.prometheus_port)
