# Project: PRAHARI — City-Scale Vision Surveillance Platform
(working name; change if you want. Project dir: `cctv_pieds`.)

You are building a production-grade multi-camera video analytics platform intended for city- and state-scale law-enforcement deployment. This is an ambitious system. Architect it correctly from day one and build in defined phases. Be opinionated. When you make a call, name the alternatives you rejected in one line each.

## High-Level Goal
A unified vision platform that:
- Ingests RTSP feeds from hundreds-to-thousands of CCTV cameras
- Runs real-time analytics on each feed (face det+recog, ANPR, person/vehicle tracking, crowd analytics, violence detection)
- Provides cross-camera identity tracking ("seen at cam 1 → again at cam 56")
- Watchlist matching against a criminal/persons-of-interest database with operator-controllable alerting
- Live operator dashboard with map, alert feed, search, and forensic replay

## Core Capabilities

### 1. Multi-Camera Ingestion
- RTSP/RTMP/HLS with per-camera config: URL, lat/lon, FOV bearing, owner agency, calibration metadata
- Stream health: FPS, bitrate, dropped-frame %, last-seen, auto-reconnect with backoff
- GPU-accelerated decode (NVDEC via PyAV; consider DeepStream for the hot path on Jetson/server GPU)
- Configurable per-analytic frame sampling (face every Nth frame, ANPR triggered by plate detector, etc.)

### 2. Face Detection + Recognition + Cross-Camera ReID
- Detector: SCRFD or YOLOv8-face (latency-optimized via TensorRT)
- 5-pt landmark alignment → 112×112
- Embedding: ArcFace or AdaFace (512-d), use a backbone trained for low-res CCTV faces (GLINT360K base, optionally fine-tuned)
- Vector DB: Milvus or Qdrant, HNSW index, cosine distance
- Quality gate: emit a face-quality score (blur, pose, size, illumination) and skip embeddings below threshold — junk embeddings poison ReID
- Cross-camera ReID: any embedding match above threshold across cameras emits a `same_identity` event linking sightings

### 3. ANPR — India-Tuned
- Plate detector: YOLOv8 fine-tuned on Indian plates (white/yellow/green/black; HSRP and non-HSRP; commercial/private/electric/BH-series)
- OCR: PaddleOCR or a CRNN trained on Indian plate fonts
- Format normalization to canonical `XX00XX0000` and BH-series; store raw + normalized
- Confidence gating: low-confidence reads emit `low_confidence_plate` instead of a wrong commit
- **Vehicle ReID**: pair plate with a vehicle-attribute classifier (type, make/model where feasible, dominant color) so a vehicle is trackable across cameras even when the plate is unreadable in one feed
- India-specific: also detect helmet absence on two-wheelers, triple-riding, wrong-way driving — these are high-value flags for traffic enforcement and trivial extensions on top of the same detector

### 4. Multi-Object Tracking
- Per-camera MOT: ByteTrack or BoT-SORT on a unified person+vehicle detector (YOLOv8 or RT-DETR)
- Track IDs are local to the camera; global identity is reconstructed by the ReID service

### 5. Watchlist / Criminal DB
Schema (Postgres):
```
subjects(subject_id, full_name, aliases[], dob, sex,
         wanted_for, priority ENUM(low|medium|high|critical),
         issuing_agency, case_ref, valid_from, valid_until,
         active BOOL, notes)
subject_faces(face_id, subject_id, image_path, embedding VECTOR(512),
              source ENUM(mugshot|cctv_capture|other), captured_at, quality_score)
subject_vehicles(...)  -- plate + vehicle attributes
```
- Multiple embeddings per subject (front, side, recent, old) — match if any exceeds threshold
- Bulk ingest from CSV + images folder
- Operator-facing toggles, exactly as specified:
  - Global watchlist alerts on/off (auto-resume timer optional)
  - Per-priority toggles (e.g., suppress `low` during festivals)
  - Per-subject snooze
  - Per-rule cooldown to prevent storms
- Every alert: thumbnail, source camera, lat/lon on map, match score, link to clip

### 6. Crowd Analytics — India-Tuned for High Density
The user specifically called this out: standard western crowd models break on Indian density. Design for that.
- **Density estimation**: CSRNet, DM-Count, or a recent transformer head (CrowdViT-style). Train/fine-tune on **dense** datasets (NWPU-Crowd, JHU-Crowd++, ShanghaiTech Part A). Note in the model card that Mall/UCSD-class data severely underestimates Indian crowds and must NOT be used as a primary training source. If festival/Kumbh footage is available locally, fine-tune on it.
- **Homography calibration per camera** (one-time setup): operator marks 4 known-distance ground points → density and velocity become real units (people/m², m/s) instead of pixels. Without this, thresholds are meaningless.
- **Flow estimation**: RAFT (accuracy) or Farneback (speed) over density maps → 2D vector field
- **Stampede signals (composite, never single-signal)**:
  - Density crossing per-camera thresholds (e.g., >5 people/m² critical)
  - Sudden flow divergence (people exploding outward from a point)
  - Sudden convergence + density spike
  - Velocity variance spike (panic indicator)
  - Counter-flow (movement against dominant direction)
- Output: per-camera crowd state + composite stampede risk score with explainability ("flagged because: divergence ↑, density 6.2, variance ↑")

### 7. Violence / Fight / Road-Rage Detection
Two-stage to keep cost sane:
- **Stage 1 (cheap, always-on)**: pose-based heuristics on tracked persons — sudden high-velocity limb movement, two tracks in close proximity with intersecting trajectories, falls. Use YOLOv8-pose or RTMPose.
- **Stage 2 (expensive, triggered)**: clip-level action classifier — VideoMAE-v2 / MViTv2 / X3D fine-tuned on RWF-2000 + Hockey Fight + Real-Life Violence Situations.
- **On confirmed event**:
  - Auto-clip ±15s
  - Extract all face embeddings + all readable plates in clip
  - Create `incident` record linking subjects + vehicles + camera + time
  - Push alert with clip URL
- **Road-rage variant**: Stage 1 looks for vehicle-vehicle close-proximity → driver/passenger exit → high-motion gestures, then Stage 2 confirms.

### 8. Alert Engine
- Rule = `(trigger_type, conditions, channels, cooldown, enabled)`
- Channels: dashboard websocket (always), SMS (MSG91/Twilio), email, optional webhook to existing CAD/ICCC
- Cooldown per `(rule, subject_or_plate, area)` to prevent storms
- All toggles surfaced in dashboard; every change is audit-logged

### 9. Search & Forensics
- Face search: upload image → all sightings within time window
- Plate search: text → all sightings (fuzzy on OCR errors)
- Vehicle attribute search: "red Maruti Swift, cam 12, 14:00–16:00"
- Incident replay: multi-cam timeline assembled around an event
- Geo-temporal queries: "all sightings of subject X in 5 km radius of point Y in last 24 h"

### 10. Dashboard (Web)
- Map view: cameras as pins, colored by health + active alerts (maplibre-gl + OSM tiles)
- Live alert feed, filterable by priority/type/area
- Per-camera live tile with detection overlays + density heatmap toggle
- Watchlist CRUD with bulk import
- Rules & toggles page (this is where the on/off switches the user described live)
- Search UI
- Incident timeline view
- Auth + RBAC: `operator | supervisor | admin | auditor`
- **Non-deletable audit log** of every search, watchlist edit, alert ack, toggle change

## Architecture

Microservices, message bus between them:

- **ingest-service** — RTSP decode → frames → bus
- **inference-service** (GPU) — detector + tracker + embeddings → events
- **reid-service** — vector-DB queries → identity-link events
- **analytics-service** (GPU) — crowd + violence (separate workers, lower FPS, heavier)
- **alert-service** — rule eval + delivery + cooldowns
- **api-service** (FastAPI) — dashboard backend, search, CRUD
- **frontend** (Next.js)

Storage:
- **Postgres + TimescaleDB** for events/sightings/alerts (hypertables on time)
- **Milvus or Qdrant** for face + vehicle embeddings
- **MinIO/S3** for frame snapshots + incident clips
- **Redis** for hot state (active tracks, recent embeddings cache, rate limits)
- **Bus**: Redis Streams initially; migrate to Kafka if scale demands. Surface tradeoffs.

## Tech Stack (defaults — change only with reason)
- Python 3.11, PyTorch + ONNX Runtime / TensorRT
- FastAPI, Pydantic v2, Uvicorn, async everywhere
- Next.js 14 + Tailwind + shadcn/ui + maplibre-gl
- uv for Python deps (pinned)
- Docker Compose for dev; k8s manifests scaffolded for prod

## Non-Negotiables
- **Deep telemetry from day one**: Prometheus metrics in every service (FPS, queue depth, inference latency p50/p95/p99, GPU mem, dropped frames, per-rule alert rate). Grafana dashboards in `infra/grafana`. Structured JSON logs with trace IDs.
- **Reproducible**: pinned deps, model weights versioned by hash, seeds where applicable
- **Tested**: pytest, sample RTSP recordings in `tests/fixtures/`, CI via GitHub Actions
- **Configurable**: every threshold, model path, rule in YAML/env — not hard-coded
- **Privacy & legal**: configurable per-data-class retention, full audit logs, redaction support for exports. Note in the README this system must be deployed under a documented legal authority — flag where that affects design (e.g., who can access raw face embeddings vs masked thumbnails).

## Phased Build — Do It in This Order

**Phase 0 — Foundations (NO ML yet)**
- Monorepo: `services/`, `frontend/`, `infra/`, `models/`, `notebooks/`, `tests/`
- Docker Compose: Postgres+TimescaleDB, Redis, Milvus, MinIO, Prometheus, Grafana
- One ingest-service pulling one RTSP, decoding, pushing frames to a Redis Stream
- Skeleton api-service + Next.js dashboard rendering the live frame
- End-to-end telemetry wired
- `docker compose up` works on a fresh clone

**Phase 1 — Single-camera detection + tracking + ANPR**
- YOLOv8 person+vehicle + ByteTrack
- Indian-format ANPR pipeline
- Sightings → Postgres, snapshots → MinIO
- Live tile with overlays in dashboard

**Phase 2 — Face pipeline + Watchlist**
- SCRFD + ArcFace → Milvus
- Watchlist CRUD + bulk import
- Watchlist hit alerting + dashboard toggles (the on/off switches)

**Phase 3 — Cross-camera ReID**
- Scale to N cameras
- ReID service for face + vehicle
- "Track this entity across cameras" UI

**Phase 4 — Crowd analytics**
- Density model + per-camera homography calibration tool
- Flow + composite stampede signals
- Heatmap overlay + crowd alerts

**Phase 5 — Violence / road-rage**
- Two-stage pipeline
- Auto-clip + face/plate extraction on incident
- Incident records linking everything

**Phase 6 — Hardening**
- Auth, RBAC, audit log
- k8s deployment, autoscaling, load test to target camera count
- Privacy controls + retention enforcement

## Hardware
Dev: RTX 3070 Ti (8 GB) + Jetson Orin available for edge experiments. Assume prod = multi-GPU servers (A6000-class) plus optional Jetson edge nodes co-located with camera clusters. Architect so the same inference container can run in either place with config-only changes.

## What I Want You to Do First
1. Read the spec, then **call out anything you would change before we write code**. Do not silently deviate — surface it.
2. Identify the 3 highest-risk technical unknowns and propose how we de-risk each (e.g., RTSP decode throughput per GPU, ReID precision/recall on real Indian CCTV, crowd-model accuracy at festival density).
3. Build Phase 0 fully. Working `docker compose up`. Hand me a checklist of what to verify.
4. List exact model weights needed for Phase 1 with sources, licenses, and SHA256s.

## Communication Style
Direct, terse, no fluff. Show code. Name rejected alternatives in one line. Use `TODO` for stubs. If something is unrealistic for the budget/timeline/hardware, say so plainly.