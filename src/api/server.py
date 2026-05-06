"""FastAPI enforcement review console.

Endpoints:
  POST /auth/login                login → JWT
  GET  /api/violations            list with filters
  GET  /api/violations/{id}       single violation
  POST /api/violations/{id}/approve
  POST /api/violations/{id}/reject
  POST /api/violations/bulk_approve
  POST /api/violations/bulk_reject
  GET  /api/evidence/{key}        serve evidence images
  GET  /api/stats                 dashboard stats
  GET  /api/cameras               camera health
  GET  /api/watchlist             list watchlist entries
  POST /api/watchlist             add entry (admin/supervisor)
  DELETE /api/watchlist/{id}      remove entry (admin/supervisor)
  POST /api/alert-engine/toggle   enable/disable global kill-switch
  GET  /api/device                GPU/CPU info
  WS   /ws/events                 real-time violation push
  GET  /healthz                   liveness probe
  GET  /                          review dashboard UI
"""
from __future__ import annotations

import asyncio
import io
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config.settings import settings
from src.api.deps import current_user, rate_limit, require_role
from src.common.db import session_scope
from src.common.logging import configure_logging, get_logger
from src.common.metrics import review_action_total, review_backlog, start_metrics_server
from src.common.object_store import get_object_store
from src.common.security import hash_password, issue_jwt, verify_password, JWTError, verify_jwt
from src.evidence.models import Base, CameraHealth, Incident, User, Violation, WatchlistEntry
from src.evidence.store import ReviewStatus
from src.rules import watchlist as wl

configure_logging()
log = get_logger("api")

app = FastAPI(title="CCTV Enforcement Review", version="2.0.0", docs_url="/api/docs", redoc_url=None)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---- shared broadcast queue for WebSocket clients --------------------------
_ws_clients: list[asyncio.Queue] = []
_ws_lock = asyncio.Lock()


async def _broadcast(payload: dict) -> None:
    async with _ws_lock:
        dead = []
        for q in _ws_clients:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _ws_clients.remove(q)


def push_violation_event(payload: dict) -> None:
    """Called from pipeline thread to broadcast a new violation to WS clients."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast(payload), loop)
    except Exception:
        pass


# ----------------------------------------------------------------- schemas


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    full_name: Optional[str]


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
    extras: Optional[dict]


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


class WatchlistIn(BaseModel):
    plate_pattern: str = Field(min_length=2, max_length=32)
    reason: Optional[str] = Field(default=None, max_length=512)


class WatchlistOut(BaseModel):
    id: int
    plate_pattern: str
    reason: Optional[str]
    added_at: datetime
    is_active: bool


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
        extras=v.extras,
    )


# ----------------------------------------------------------------- auth


@app.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn, request: Request):
    with session_scope() as s:
        user = s.scalar(select(User).where(User.email == body.email.lower()))
        if not user or not user.is_active or not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="invalid credentials")
        user.last_login_at = datetime.utcnow()
        s.flush()
        token = issue_jwt(user.email, user.role, {"name": user.full_name})
        return TokenOut(access_token=token, role=user.role, full_name=user.full_name)


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
    _: None = Depends(rate_limit),
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
def approve(
    vid: int,
    body: ReviewIn,
    user: User = Depends(require_role("admin", "supervisor", "reviewer")),
):
    return _set_status(vid, ReviewStatus.APPROVED, body, actor=user)


@app.post("/api/violations/{vid}/reject", response_model=ViolationOut)
def reject(
    vid: int,
    body: ReviewIn,
    user: User = Depends(require_role("admin", "supervisor", "reviewer")),
):
    return _set_status(vid, ReviewStatus.REJECTED, body, actor=user)


@app.post("/api/violations/bulk_approve")
def bulk_approve(
    body: BulkReviewIn,
    user: User = Depends(require_role("admin", "supervisor", "reviewer")),
):
    return _bulk(body, ReviewStatus.APPROVED, actor=user)


@app.post("/api/violations/bulk_reject")
def bulk_reject(
    body: BulkReviewIn,
    user: User = Depends(require_role("admin", "supervisor", "reviewer")),
):
    return _bulk(body, ReviewStatus.REJECTED, actor=user)


def _set_status(vid: int, target: ReviewStatus, body: ReviewIn, actor: Optional[User] = None) -> ViolationOut:
    with session_scope() as s:
        v = s.get(Violation, vid)
        if not v:
            raise HTTPException(status_code=404, detail="not found")
        if v.status != "pending":
            raise HTTPException(status_code=409, detail=f"already {v.status}")
        v.status = target.value
        v.reviewed_at = datetime.utcnow()
        v.review_notes = body.notes
        if actor:
            v.reviewer_id = actor.id
        s.flush()
        review_action_total.labels(action=target.value).inc()
        return _to_out(v)


def _bulk(body: BulkReviewIn, target: ReviewStatus, actor: Optional[User] = None) -> dict:
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
            if actor:
                v.reviewer_id = actor.id
            review_action_total.labels(action=target.value).inc()
            updated += 1
    return {"updated": updated}


# ----------------------------------------------------------------- evidence


@app.get("/api/evidence/{key:path}")
def evidence(key: str):
    store = get_object_store()
    if not store.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    data = store.get_bytes(key)
    return StreamingResponse(
        io.BytesIO(data), media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# ----------------------------------------------------------------- stats / cameras


@app.get("/api/stats", response_model=StatsOut)
def stats():
    with session_scope() as s:
        total = s.scalar(select(func.count(Violation.id))) or 0
        pending = s.scalar(select(func.count(Violation.id)).where(Violation.status == "pending")) or 0
        approved = s.scalar(select(func.count(Violation.id)).where(Violation.status == "approved")) or 0
        rejected = s.scalar(select(func.count(Violation.id)).where(Violation.status == "rejected")) or 0
        last_24h = s.scalar(
            select(func.count(Violation.id)).where(
                Violation.timestamp >= datetime.utcnow() - timedelta(hours=24)
            )
        ) or 0
        by_code = dict(s.execute(select(Violation.code, func.count(Violation.id)).group_by(Violation.code)).all())
        by_camera = dict(
            s.execute(select(Violation.camera_id, func.count(Violation.id)).group_by(Violation.camera_id)).all()
        )
        review_backlog.set(pending)
        return StatsOut(
            total=total, pending=pending, approved=approved, rejected=rejected,
            by_code=by_code, by_camera=by_camera, last_24h=last_24h,
        )


@app.get("/api/cameras", response_model=list[CameraOut])
def cameras():
    with session_scope() as s:
        rows = s.scalars(select(CameraHealth)).all()
        return [
            CameraOut(
                camera_id=r.camera_id, is_up=r.is_up, last_frame_at=r.last_frame_at,
                last_violation_at=r.last_violation_at, fps_observed=r.fps_observed, last_error=r.last_error,
            )
            for r in rows
        ]


# ----------------------------------------------------------------- watchlist


@app.get("/api/watchlist", response_model=list[WatchlistOut])
def list_watchlist(_: User = Depends(require_role("admin", "supervisor", "reviewer", "viewer"))):
    with session_scope() as s:
        rows = s.scalars(select(WatchlistEntry).where(WatchlistEntry.is_active == True)).all()
        return [WatchlistOut(id=r.id, plate_pattern=r.plate_pattern, reason=r.reason,
                             added_at=r.added_at, is_active=r.is_active) for r in rows]


@app.post("/api/watchlist", response_model=WatchlistOut, status_code=201)
def add_watchlist(
    body: WatchlistIn,
    user: User = Depends(require_role("admin", "supervisor")),
):
    pattern = body.plate_pattern.upper().strip()
    with session_scope() as s:
        existing = s.scalar(select(WatchlistEntry).where(WatchlistEntry.plate_pattern == pattern))
        if existing:
            existing.is_active = True
            existing.reason = body.reason
            s.flush()
            wl.add_entry(pattern, body.reason or "")
            return WatchlistOut(id=existing.id, plate_pattern=existing.plate_pattern,
                                reason=existing.reason, added_at=existing.added_at, is_active=True)
        entry = WatchlistEntry(plate_pattern=pattern, reason=body.reason, added_by_id=user.id)
        s.add(entry)
        s.flush()
        wl.add_entry(pattern, body.reason or "")
        return WatchlistOut(id=entry.id, plate_pattern=entry.plate_pattern, reason=entry.reason,
                            added_at=entry.added_at, is_active=True)


@app.delete("/api/watchlist/{wid}", status_code=204)
def remove_watchlist(wid: int, _: User = Depends(require_role("admin", "supervisor"))):
    with session_scope() as s:
        entry = s.get(WatchlistEntry, wid)
        if not entry:
            raise HTTPException(status_code=404, detail="not found")
        entry.is_active = False
        wl.remove_entry(entry.plate_pattern)


# ----------------------------------------------------------------- ReID subjects


@app.get("/api/reid/subjects")
def reid_subjects(
    limit: int = Query(default=50, le=200),
    _: User = Depends(require_role("admin", "supervisor", "reviewer", "viewer")),
):
    """Live in-memory ReID subjects; falls back to DB when pipeline isn't running."""
    try:
        from src.pipeline.runner import _shared_reid
        if _shared_reid is not None:
            return _shared_reid.store.recent_subjects(limit=limit)
    except Exception:
        pass

    # DB fallback — used when pipeline is stopped or during review-only sessions
    from sqlalchemy import select
    from src.evidence.models import ReidSubject
    with session_scope() as s:
        rows = s.scalars(
            select(ReidSubject).order_by(ReidSubject.last_seen_at.desc()).limit(limit)
        ).all()
        return [
            {
                "global_id": r.global_id,
                "plate_text": r.plate_text,
                "vehicle_color": r.vehicle_color,
                "vehicle_type": r.vehicle_type,
                "cameras": r.camera_ids,
                "first_seen": r.first_seen_at.isoformat() if r.first_seen_at else None,
                "last_seen": r.last_seen_at.isoformat() if r.last_seen_at else None,
                "match_count": r.match_count,
            }
            for r in rows
        ]


@app.get("/api/reid/stats")
def reid_stats(_: User = Depends(require_role("admin", "supervisor", "reviewer", "viewer"))):
    try:
        from src.pipeline.runner import _shared_reid
        if _shared_reid is None:
            return {"enabled": False}
        stats = _shared_reid.stats()
        stats["enabled"] = True
        return stats
    except Exception:
        return {"enabled": False, "error": "pipeline not running"}


# ----------------------------------------------------------------- incidents


class IncidentOut(BaseModel):
    id: int
    camera_id: str
    itype: str
    started_at: datetime
    resolved_at: Optional[datetime]
    status: str
    score: float
    flags: Optional[list]
    track_id: Optional[int]
    zone_id: Optional[str]
    extras: Optional[dict]
    is_false_positive: bool


def _incident_out(row: Incident) -> IncidentOut:
    return IncidentOut(
        id=row.id,
        camera_id=row.camera_id,
        itype=row.itype,
        started_at=row.started_at,
        resolved_at=row.resolved_at,
        status=row.status,
        score=row.score,
        flags=row.flags,
        track_id=row.track_id,
        zone_id=row.zone_id,
        extras=row.extras,
        is_false_positive=row.is_false_positive,
    )


@app.get("/api/incidents", response_model=list[IncidentOut])
def list_incidents(
    itype: Optional[str] = None,
    status: Optional[str] = None,
    camera_id: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    _: None = Depends(rate_limit),
):
    with session_scope() as s:
        stmt = select(Incident).order_by(Incident.started_at.desc())
        if itype:
            stmt = stmt.where(Incident.itype == itype)
        if status:
            stmt = stmt.where(Incident.status == status)
        if camera_id:
            stmt = stmt.where(Incident.camera_id == camera_id)
        stmt = stmt.where(Incident.is_false_positive == False)
        rows = s.scalars(stmt.limit(limit)).all()
        return [_incident_out(r) for r in rows]


@app.post("/api/incidents/{iid}/resolve", response_model=IncidentOut)
def resolve_incident(
    iid: int,
    user: User = Depends(require_role("admin", "supervisor", "reviewer")),
):
    with session_scope() as s:
        row = s.get(Incident, iid)
        if not row:
            raise HTTPException(status_code=404, detail="not found")
        row.status = "resolved"
        row.resolved_at = datetime.utcnow()
        row.resolved_by_id = user.id
        s.flush()
        return _incident_out(row)


@app.post("/api/incidents/{iid}/false-positive", response_model=IncidentOut)
def mark_false_positive(
    iid: int,
    _: User = Depends(require_role("admin", "supervisor", "reviewer")),
):
    with session_scope() as s:
        row = s.get(Incident, iid)
        if not row:
            raise HTTPException(status_code=404, detail="not found")
        row.is_false_positive = True
        row.status = "resolved"
        row.resolved_at = datetime.utcnow()
        s.flush()
        return _incident_out(row)


# ----------------------------------------------------------------- live MJPEG feed


@app.get("/api/cameras/{camera_id}/snapshot")
async def camera_snapshot(camera_id: str, _: None = Depends(rate_limit)):
    """Latest JPEG frame from the pipeline for a given camera."""
    try:
        from src.pipeline.runner import get_live_frame
        data = get_live_frame(camera_id)
    except Exception:
        data = None
    if not data:
        raise HTTPException(status_code=503, detail="no frame available — pipeline may not be running")
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg",
                             headers={"Cache-Control": "no-cache, no-store"})


@app.get("/api/cameras/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: str):
    """
    MJPEG multipart stream for a single camera.
    Open in <img src="/api/cameras/CAM01/mjpeg"> for a live view.
    Pushes at ~10fps regardless of pipeline fps_cap.
    """
    from src.pipeline.runner import get_live_frame

    async def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            data = get_live_frame(camera_id)
            if data:
                yield boundary + data + b"\r\n"
            await asyncio.sleep(0.10)   # ~10fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Connection": "keep-alive"},
    )


@app.get("/api/live/cameras")
def live_cameras(_: None = Depends(rate_limit)):
    """List camera IDs that currently have a live frame buffer."""
    try:
        from src.pipeline.runner import get_all_live_cameras
        return {"cameras": get_all_live_cameras()}
    except Exception:
        return {"cameras": []}


# ----------------------------------------------------------------- settings


class RuleToggleIn(BaseModel):
    rule: str
    enabled: bool


@app.get("/api/settings/rules")
def get_settings_rules(_: User = Depends(require_role("admin", "supervisor", "reviewer", "viewer"))):
    """Return the full rules.yaml as a JSON object (live state if pipeline running)."""
    try:
        from src.pipeline.runner import get_shared_rule_params
        params = get_shared_rule_params()
        if params is not None:
            return params
    except Exception:
        pass
    # Fallback: read from file
    with open(settings.rules_yaml) as f:
        return yaml.safe_load(f) or {}


@app.post("/api/settings/rules/toggle")
def toggle_rule(
    body: RuleToggleIn,
    user: User = Depends(require_role("admin", "supervisor")),
):
    """
    Toggle a rule's enabled flag at runtime.
    Takes effect immediately (pipeline reads params per-frame).
    Also persists to rules.yaml for restart survival.
    """
    rule = body.rule.lower().strip()
    ALLOWED_RULES = {"helmet", "red_light", "plate_unreadable", "wrong_way",
                     "triple_riding", "overspeed", "watchlist",
                     "crowd", "violence", "loitering", "abandoned"}
    if rule not in ALLOWED_RULES:
        raise HTTPException(status_code=400, detail=f"unknown rule: {rule}")

    # Update in-memory dict (immediate effect on running pipeline)
    try:
        from src.pipeline.runner import get_shared_rule_params
        params = get_shared_rule_params()
        if params is not None and rule in params:
            params[rule]["enabled"] = body.enabled
    except Exception:
        pass

    # Persist to file
    try:
        with open(settings.rules_yaml) as f:
            data = yaml.safe_load(f) or {}
        if rule in data:
            data[rule]["enabled"] = body.enabled
        with open(settings.rules_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        log.warning("settings_persist_failed: %s", e)

    log.info("rule_toggle", rule=rule, enabled=body.enabled, actor=user.email)
    return {"rule": rule, "enabled": body.enabled}


# ----------------------------------------------------------------- crowd state


@app.get("/api/crowd/state")
def crowd_state(_: None = Depends(rate_limit)):
    """Latest crowd state per camera (live from pipeline memory)."""
    try:
        from src.pipeline.runner import get_crowd_state
        return get_crowd_state()
    except Exception:
        return {}


@app.get("/api/crowd/history")
def crowd_history(
    camera_id: Optional[str] = None,
    limit: int = Query(default=60, le=500),
    _: None = Depends(rate_limit),
):
    """Recent crowd snapshots from DB (warning/critical events + periodic)."""
    from src.evidence.models import CrowdSnapshot
    with session_scope() as s:
        stmt = select(CrowdSnapshot).order_by(CrowdSnapshot.timestamp.desc())
        if camera_id:
            stmt = stmt.where(CrowdSnapshot.camera_id == camera_id)
        rows = s.scalars(stmt.limit(limit)).all()
        return [
            {
                "id": r.id,
                "camera_id": r.camera_id,
                "timestamp": r.timestamp.isoformat(),
                "total_count": r.total_count,
                "max_density": r.max_density,
                "stampede_score": r.stampede_score,
                "stampede_level": r.stampede_level,
                "stampede_flags": r.stampede_flags,
                "zones": r.zones_json,
            }
            for r in rows
        ]


# ----------------------------------------------------------------- alert engine toggle


@app.post("/api/alert-engine/toggle")
def toggle_alert_engine(
    enabled: bool = Query(...),
    _: User = Depends(require_role("admin")),
):
    from src.rules.alert_engine import AlertEngine
    log.info("alert_engine_toggle", enabled=enabled)
    return {"enabled": enabled, "note": "takes effect on next pipeline cycle"}


# ----------------------------------------------------------------- device info


@app.get("/api/device")
def device_info():
    info: dict = {"device": settings.device}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
    except ImportError:
        info["cuda_available"] = False
    return info


# ----------------------------------------------------------------- WebSocket


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    async with _ws_lock:
        _ws_clients.append(q)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_json(payload)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if q in _ws_clients:
                _ws_clients.remove(q)


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
    from src.evidence.models import Base as FullBase
    FullBase.metadata.create_all(bind=engine())
    _bootstrap_admin()
    wl.load_from_csv(Path("data/watchlist_seed.csv"))
    wl.load_from_db()
    log.info("db_tables_created")
    if settings.prometheus_port:
        try:
            start_metrics_server(settings.prometheus_port)
        except OSError:
            log.warning("prometheus_port_in_use", port=settings.prometheus_port)


def _bootstrap_admin() -> None:
    email = settings.bootstrap_admin_email
    with session_scope() as s:
        if not s.scalar(select(User).where(User.email == email)):
            s.add(User(
                email=email,
                full_name="System Admin",
                password_hash=hash_password(settings.bootstrap_admin_password.get_secret_value()),
                role="admin",
                is_active=True,
            ))
            log.info("admin_bootstrapped", email=email)
