"""SQLAlchemy ORM models — production schema.

Three tables:
  - users           operator/admin/auditor accounts
  - violations      one row per fired event (immutable except for review fields)
  - audit_log       append-only record of every state change

Indexes are tuned for the operator dashboard's most common queries:
  - "show pending violations newest first"
  - "lookup by plate"
  - "filter by camera & day"
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    badge_number = Column(String(64), nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="operator")  # admin | operator | auditor
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32), index=True, nullable=False)
    camera_id = Column(String(64), index=True, nullable=False)
    track_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    frame_idx = Column(Integer, nullable=False)

    plate_text = Column(String(32), index=True, nullable=True)
    plate_ocr_confidence = Column(Float, nullable=True)
    rule_confidence = Column(Float, nullable=False)

    # Asset locations — may be local paths under EVIDENCE_DIR
    # or s3 keys under S3_BUCKET, depending on EVIDENCE_BACKEND.
    frame_path = Column(String(512), nullable=False)
    annotated_path = Column(String(512), nullable=True)
    plate_crop_path = Column(String(512), nullable=True)

    bbox = Column(JSON, nullable=True)
    extras = Column(JSON, nullable=True)

    # Tamper-evident chain.
    sha256 = Column(String(64), nullable=False)        # hash of frame bytes
    signature = Column(LargeBinary, nullable=False)    # Ed25519(canonical_json + sha256)
    prev_id = Column(Integer, nullable=True)           # links to previous violation for audit chain
    chain_hash = Column(String(64), nullable=False)    # sha256(prev_chain_hash || this.sha256)

    # Review workflow
    status = Column(String(16), default="pending", index=True, nullable=False)
    reviewer_id = Column(Integer, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)

    # Chalana push state
    pushed_to_chalana = Column(Boolean, nullable=False, default=False)
    pushed_at = Column(DateTime, nullable=True)
    chalana_reference = Column(String(128), nullable=True)

    __table_args__ = (
        Index("ix_violations_status_timestamp", "status", "timestamp"),
        Index("ix_violations_camera_timestamp", "camera_id", "timestamp"),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    actor_id = Column(Integer, nullable=True)              # user id (null for system actions)
    actor_label = Column(String(128), nullable=True)       # "system" | email | service name
    action = Column(String(64), index=True, nullable=False)
    target_type = Column(String(64), nullable=True)
    target_id = Column(String(64), nullable=True)
    detail = Column(JSON, nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)


class CameraHealth(Base):
    __tablename__ = "camera_health"

    camera_id = Column(String(64), primary_key=True)
    last_frame_at = Column(DateTime, nullable=True)
    last_violation_at = Column(DateTime, nullable=True)
    fps_observed = Column(Float, nullable=True)
    is_up = Column(Boolean, nullable=False, default=False)
    last_error = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class WatchlistEntry(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_pattern = Column(String(32), unique=True, index=True, nullable=False)
    reason = Column(String(512), nullable=True)
    added_by_id = Column(Integer, nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)


class ReidSubject(Base):
    """Persistent cross-camera identity — one row per unique vehicle seen across cameras."""
    __tablename__ = "reid_subjects"

    global_id = Column(String(36), primary_key=True)   # UUID
    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    camera_ids = Column(JSON, nullable=False, default=list)    # list[str]
    plate_text = Column(String(32), nullable=True, index=True)
    vehicle_color = Column(String(32), nullable=True)
    vehicle_type = Column(String(32), nullable=True)
    match_count = Column(Integer, nullable=False, default=1)   # total cross-camera matches
    is_watchlisted = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_reid_subjects_plate", "plate_text"),
        Index("ix_reid_subjects_last_seen", "last_seen_at"),
    )


class Incident(Base):
    """Safety incident — violence, loitering, stampede, abandoned object."""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), index=True, nullable=False)
    # violence | loitering | abandoned | stampede
    itype = Column(String(32), index=True, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    # suspected → active → resolved
    status = Column(String(16), nullable=False, default="suspected", index=True)
    score = Column(Float, nullable=False, default=0.0)
    flags = Column(JSON, nullable=True)
    track_id = Column(Integer, nullable=True)
    zone_id = Column(String(64), nullable=True)
    extras = Column(JSON, nullable=True)
    is_false_positive = Column(Boolean, nullable=False, default=False)
    resolved_by_id = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_incidents_camera_status", "camera_id", "status"),
        Index("ix_incidents_type_started", "itype", "started_at"),
    )


class PlateSighting(Base):
    """Surveillance plate log — every plate seen on every camera, every time.

    The mass-surveillance backbone: cross-camera matching, suspect tracking,
    and historic lookups all read from this table. Write rate is one row per
    unique (camera_id, track_id, plate_text) triple — no per-frame spam.
    """
    __tablename__ = "plate_sightings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_text = Column(String(32), index=True, nullable=False)
    camera_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    track_id = Column(Integer, nullable=True)
    global_id = Column(String(36), nullable=True, index=True)   # ReID global id if assigned
    ocr_confidence = Column(Float, nullable=False, default=0.0)
    vehicle_class = Column(String(32), nullable=True)            # car | two_wheeler | bus | truck
    vehicle_color = Column(String(32), nullable=True)
    plate_crop_path = Column(String(512), nullable=True)         # cropped plate image
    is_cross_camera_match = Column(Boolean, nullable=False, default=False, index=True)

    __table_args__ = (
        Index("ix_plate_sightings_plate_ts", "plate_text", "timestamp"),
        Index("ix_plate_sightings_camera_ts", "camera_id", "timestamp"),
        # Fast lookup for the per-track UPDATE path (every plate read
        # does s.get(PlateSighting, existing_id) — that's already O(1)
        # by primary key, but the per-track ordering query in the
        # face-capture replace path is hit hardest):
        Index("ix_plate_sightings_camera_track", "camera_id", "track_id"),
    )


class FaceCapture(Base):
    """Face captured during a violence incident.

    Stored with a 384-d DINOv2 embedding so the police-DB matcher can do
    a cosine-similarity lookup; the matched record id + similarity are
    stamped on this row when a hit is found.
    """
    __tablename__ = "face_captures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, nullable=True, index=True)
    camera_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    track_id = Column(Integer, nullable=True)
    face_crop_path = Column(String(512), nullable=False)
    embedding = Column(LargeBinary, nullable=True)              # 384-d float32 .tobytes()
    quality_score = Column(Float, nullable=False, default=0.0)  # crop confidence / sharpness

    # Mock police-DB match (replace with real provider in production)
    matched_record_id = Column(String(64), nullable=True, index=True)
    matched_name = Column(String(255), nullable=True)
    matched_charges = Column(Text, nullable=True)
    matched_risk_level = Column(String(16), nullable=True)
    match_similarity = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_face_captures_incident", "incident_id"),
        Index("ix_face_captures_match", "matched_record_id"),
    )


class CrowdSnapshot(Base):
    """Point-in-time crowd density snapshot — written on warning/critical or every N frames."""
    __tablename__ = "crowd_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    total_count = Column(Integer, nullable=False, default=0)
    max_density = Column(Float, nullable=False, default=0.0)
    stampede_score = Column(Float, nullable=False, default=0.0)
    stampede_level = Column(String(16), nullable=False, default="ok")
    stampede_flags = Column(JSON, nullable=True)
    zones_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_crowd_snapshots_camera_ts", "camera_id", "timestamp"),
    )
