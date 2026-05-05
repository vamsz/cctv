# 4-Step Build Plan — Police-Grade CCTV Enforcement (India)

## Current State
The friend's code gives us: RTSP ingest → YOLO11 detect → ByteTrack → PaddleOCR → rules (helmet, red-light, plate-unreadable) → SQLite evidence → FastAPI review console.

What's missing for real police use: wrong-way/triple-riding/speed, cross-camera ReID, watchlist, crowd safety, violence detection, proper auth, and a real-time modern UI.

---

## Step 1 — Core Traffic Rules + Alert Engine + RBAC
_Test gate: Run demo video, confirm all 6 violation types fire, UI shows them, RBAC login works._

### What we build
1. **New violation rules** (add to rules engine):
   - Wrong-way driving (dot-product of velocity vs lane direction vector, per camera)
   - Triple riding (count persons overlapping upper 60% of two-wheeler bounding box ≥ 3)
   - Overspeed estimate (pixel-to-real via per-camera homography scalar, from calibrate.py)
   - Watchlist plate match (plate_text against a pre-loaded CSV/DB list → WATCHLIST_HIT)

2. **Alert rule engine hardening**:
   - Per-rule enable/disable toggle (rules.yaml already has it, enforce it everywhere)
   - Per-subject cooldowns: same plate + same rule = no repeated fire for N seconds
   - Per-priority cooldowns: HIGH=30s, MEDIUM=60s, LOW=120s global defaults
   - Global alert kill-switch (emergency mute all)

3. **RBAC + JWT auth**:
   - Roles: admin, supervisor, reviewer, viewer
   - Login endpoint → JWT token (bcrypt passwords)
   - Role-gated API endpoints (viewer can't approve)
   - Session audit log in DB (who did what when)

4. **Chalana push queue stub** (outbound worker already in codebase, wire it up properly):
   - Approved violations → outbound queue
   - POST to configurable endpoint with retry backoff

5. **GPU/CPU auto-detect fix**:
   - Auto-select cuda:0 if CUDA available, fallback to cpu
   - NVDEC concurrent stream benchmark script

---

## Step 2 — Cross-Camera ReID + Vehicle Attributes + Watchlist
_Test gate: Same vehicle tracked with consistent identity across 2 camera feeds. Watchlist hit fires._

### What we build
1. **Vehicle ReID**:
   - Lightweight OSNet embedding extractor (512-dim) from torchreid
   - FAISS IndexFlatIP vector store per session (no external DB needed locally)
   - Cross-camera matching: new vehicle embedding → nearest-neighbor search → if cosine sim > 0.85 → same vehicle
   - ReID confidence thresholding with distance decay over time

2. **Vehicle attribute classifier** (fallback when plate unreadable):
   - Single-head classifier on vehicle crop: color (12 bins), type (car/truck/bus/bike/auto), make
   - Trained on VERI-776 + scraped Indian traffic frames
   - Fallback record stored with violation when OCR fails

3. **Watchlist system**:
   - Admin API to add/remove plates and/or embeddings to watchlist
   - Every plate read checked against watchlist → WATCHLIST_HIT alert (bypasses normal cooldowns)
   - ReID cross-match against watchlist embeddings
   - UI shows watchlist hits with red priority banner

4. **Better plate detection with SAHI** (Sliced Inference for small/far plates):
   - Wrap plate model in SAHI slicer (4-slice grid)
   - ~20% accuracy lift on distant vehicles

---

## Step 3 — Crowd Safety + Anomaly Detection
_Test gate: Synthetic crowd video with increasing density triggers YELLOW then RED alert. Pose-based fighting triggers Stage-2 video model._

### What we build
1. **Crowd density with real homography**:
   - Extend calibrate.py: user marks 4 ground-plane reference points → compute H matrix
   - Person count / m² (not pixels) using homography projection
   - Density grid heatmap overlay on MJPEG feed
   - Based on JHU-Crowd++ or NWPU density models

2. **Stampede prediction (composite signal)**:
   - Optical flow (Farneback) → crowd velocity vector + variance
   - Pressure index = density × velocity_toward_bottleneck
   - Thresholds: YELLOW (density > 3/m² + rising 20%/30s), RED (density > 5/m² + exit flow blocked)
   - Never single-signal — all 3 inputs must agree before RED fires

3. **Violence / abnormal behavior (two-stage)**:
   - Stage 1 (always on, ~2ms): MediaPipe pose keypoints → heuristic: elbow velocity, arm angle vs torso → fighting score
   - Stage 2 (triggered, ~200ms): 2-second clip → SlowFast-R50 or VideoMAE inference → VIOLENCE_DETECTED alert
   - Stage 2 runs only when Stage 1 score > threshold → GPU budget preserved (≤5% frames)

4. **Loitering + abandoned object**:
   - Per-track dwell timer: person stationary at sensitive zone > cfg seconds → LOITERING
   - Object appears in frame with no person nearby for > N seconds → ABANDONED_OBJECT

---

## Step 4 — Modern UI + Production Hardening
_Test gate: Docker compose up, open localhost:8000, all pages render, WebSocket pushes violation card in real-time within 2s of detection._

### What we build
1. **Full UI rebuild** (vanilla JS, no framework, dark monochromatic):
   - Dashboard: live stats bar, real-time violation cards (WebSocket push, no polling)
   - Live feeds page: MJPEG thumbnail grid per camera, crowd density overlay
   - Analytics: violations/hour chart (Chart.js), top offenders list, camera heatmap
   - Watchlist manager: add/remove plates and flagged vehicle profiles
   - Settings: rule toggles, cooldown sliders, threshold dials per camera
   - Keyboard shortcuts: A=approve, R=reject, arrow to next, space to zoom image
   - Color language: monochrome grays + only 3 accent colors (white=neutral, amber=warning, red=violation)

2. **WebSocket real-time push**:
   - FastAPI /ws/events endpoint
   - Pipeline → asyncio queue → WebSocket broadcast
   - UI card appears within 2s of detection (no poll)

3. **Docker + GPU production**:
   - docker-compose.yml with NVIDIA runtime, camera groups per GPU
   - Multi-GPU distribution (camera groups assigned round-robin)
   - Health check endpoints wired to compose healthcheck
   - Prometheus + Grafana dashboard template (violations/s, GPU utilization, queue depth)

4. **Performance**:
   - NVDEC concurrent stream benchmark (determine max cameras for RTX 3070Ti)
   - TensorRT export automation for all 3 model heads
   - Frame-skip governor: if inference queue backs up > 3 frames, skip and log

5. **HOW TO RUN** doc (after all code done — brief, not a wall of text)

---

## Rules for us to follow
- Each step fully tested before the next begins
- No features from the next step creep into the current one
- All accuracy decisions tuned for India: mixed scripts, crowded roads, weather haze, night IR cameras
- GPU code paths always have CPU fallback
- Evidence chain (SHA-256) never broken — every new violation type goes through the same store
