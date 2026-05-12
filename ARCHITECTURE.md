# SENTINEL CCTV System — Full Architecture & Engineering Document

## What This System Is

SENTINEL is a police-grade, real-time CCTV enforcement platform for Indian traffic law. It runs on a single Windows machine with an NVIDIA RTX 3070 Ti (CUDA 12). It ingests up to 5 simultaneous camera feeds, runs a full computer-vision inference pipeline on each, detects traffic violations, captures evidence, and serves a live review console to enforcement officers through a browser.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser (officer's workstation)                                     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  index.html  (Single Page App — vanilla JS, no framework)     │  │
│  │                                                                │  │
│  │  Live Feeds tab  → WebSocket /ws/video/{cam}  (binary JPEG)   │  │
│  │  Violations tab  → REST API + WebSocket /ws/events (JSON)     │  │
│  │  Plates tab      → REST API  /api/plates                      │  │
│  │  Incidents tab   → REST API  /api/incidents                   │  │
│  │  ReID tab        → REST API  /api/reid/subjects               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
           │ HTTP / WS over localhost
┌──────────▼───────────────────────────────────────────────────────────┐
│  FastAPI server  (src/api/server.py)                                 │
│                                                                      │
│  REST endpoints for CRUD on violations, plates, faces, incidents     │
│  /ws/events  → broadcast JSON violation events to all officers       │
│  /ws/video/{camera_id}  → binary JPEG stream (30fps cap)            │
│  /api/cameras/{id}/mjpeg  → legacy MJPEG stream (kept for fallback) │
└──────────────────────────────────────────────────────────────────────┘
           │ in-process calls (same Python process)
┌──────────▼───────────────────────────────────────────────────────────┐
│  PipelineOrchestrator  (src/pipeline/runner.py)                      │
│                                                                      │
│  One CameraPipeline per camera (up to 5 simultaneous)               │
│  Shared GPU resources: Detector (YOLO), OCR, ReID, CLIP             │
│  Shared in-memory stores: _live_frames, _crowd_state, _shared_reid   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Per-Camera Threading Model

Each `CameraPipeline` runs exactly **4 threads** plus background executor pools. The fundamental principle is: **display is never blocked by inference, and inference is never blocked by I/O**.

```
Camera (RTSP / file / webcam)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  StreamReader thread  (src/ingest/stream.py)        │
│  Priority: ABOVE_NORMAL                             │
│  • Runs cv2.VideoCapture in a dedicated thread      │
│  • Always discards older frames in favour of latest │
│  • Never blocks; latest frame sits in a slot        │
└─────────────────────┬───────────────────────────────┘
                      │ reader.read() — non-blocking
                      ▼
┌─────────────────────────────────────────────────────┐
│  Buffer-fill thread  (_buffer_fill_loop)            │
│  Priority: ABOVE_NORMAL                             │
│  • Polls reader.read() at high rate                 │
│  • Pushes every new frame into FrameBuffer (300-    │
│    frame ring buffer backed by collections.deque)   │
│  • Loop sleep: 5ms  →  max ~200 reads/s            │
└──────┬──────────────────────────┬───────────────────┘
       │ .latest()                │ .pop_inference()
       ▼                          ▼
┌──────────────────┐   ┌─────────────────────────────┐
│  Stream thread   │   │  Inference thread           │
│  (_stream_loop)  │   │  (_inference_loop)          │
│  ABOVE_NORMAL    │   │  BELOW_NORMAL               │
│                  │   │                             │
│  • Reads latest  │   │  • Reads one new frame      │
│    frame from    │   │    per 100ms (10fps cap)    │
│    FrameBuffer   │   │  • Runs ALL models in seq.  │
│  • Encodes JPEG  │   │  • GPU: YOLO detect, CLIP   │
│  • Stores in     │   │  • CPU: plate OCR, pose     │
│    _live_frames  │   │  • Fires violation events   │
│  • 30fps ceiling │   │  • Persists evidence to DB  │
└──────────────────┘   └─────────────────────────────┘
       │                          │
       │ get_live_frame()         │ push_violation_event()
       ▼                          ▼
  /ws/video/{cam}            /ws/events
  WebSocket endpoint         WebSocket broadcast
```

### FrameBuffer

`FrameBuffer` is a `deque(maxlen=300)` — a ring buffer that holds up to 300 frames (~10 seconds at 30fps). Key properties:

- **`put(idx, ts, frame)`** — writer side (buffer-fill thread). Thread-safe.
- **`latest()`** — returns the newest frame without removing it. Used by stream thread. Never blocks inference.
- **`pop_inference()`** — returns the newest frame ONLY IF it hasn't been consumed by inference yet. Returns `None` if inference already saw this frame index. This prevents inference from ever re-processing the same frame, and means it naturally skips frames it can't keep up with without any explicit dropping logic.

### OS Thread Priorities

On Windows, `SetThreadPriority()` is called:
- Buffer-fill thread: `THREAD_PRIORITY_ABOVE_NORMAL` (+1)
- Stream thread: `THREAD_PRIORITY_ABOVE_NORMAL` (+1)
- Inference thread: `THREAD_PRIORITY_BELOW_NORMAL` (-1)

This means the OS scheduler guarantees the camera reader and live display are never preempted by GPU inference. On POSIX systems the equivalent `os.nice()` call is used instead.

---

## The Inference Pipeline — What Runs on Every Frame

The inference thread calls `_process_frame(frame_idx, ts, frame)` for each new frame. Here is the exact execution order:

### 1. Drain async results (zero GPU time)
At the very start of each frame, results from background executors are drained back into inference-thread state:
- **`_drain_reid_results()`** — collect completed DINOv2 embedding jobs from `_reid_executor`
- OCR results — collect completed PaddleOCR jobs from `_ocr_executor`
- DB insert results — collect completed plate sighting row IDs from `_db_executor`

### 2. Object detection (GPU)
`Detector.detect(frame)` runs YOLOv8 on the GPU. Returns `Detection` objects with class, confidence, and bounding box. Classes detected: cars, trucks, buses, motorcycles, bicycles, persons, traffic signals.

### 3. Multi-object tracking
ByteTrack tracker consumes detections and maintains consistent `track_id` across frames. Returns `TrackedDetection` objects. Track IDs are stable within a camera session.

### 4. Per-track processing loop
For each tracked vehicle/person:

**a. Signal state** — classify traffic light ROI (green/red/yellow/unknown)

**b. Plate detection + fast-ALPR** — fast-alpr (ONNX, runs synchronously, <10ms) detects and reads the licence plate region. Minimum confidence thresholds filter garbage reads:
  - Generic plates: min confidence 0.80
  - UK-format plates: rejected entirely (irrelevant for India)

**c. Async PaddleOCR (PP-OCRv5)** — submitted to `_ocr_executor` (1 worker per camera). PP-OCRv5 is a heavier, more accurate OCR engine. Results arrive on the next frame(s) and are drained at step 1. Characters are normalised to Indian plate format.

**d. PlateConsensusEngine** — multi-frame character-level voting. For each position in the plate string, keeps a counter of what character each OCR run produced. After `min_reads_for_consensus=2` reads, the majority-vote string is used. This gets ~99% per-character accuracy even with single-frame noise.

**e. Vehicle attributes** — colour (white/black/silver/red/etc.) and type (sedan/SUV/truck/etc.) classified from the vehicle crop.

**f. Async ReID (DINOv2)** — cross-camera vehicle re-identification:
  - Every 15 frames per track, a DINOv2-Small embedding is extracted asynchronously in `_reid_executor`
  - The embedding is searched against `ReIDStore` for matches from other cameras
  - If a match is found, a shared `global_id` (UUID) is assigned so the same vehicle is linked across cameras
  - If the matched entry has a plate text and we don't, it is propagated back

**g. Rules engine** — `RulesEngine.evaluate(bundle)` checks all enabled violation rules:
  - **Red-light running** — vehicle crosses stop line while signal is red
  - **Wrong way** — vehicle direction vector opposes configured flow direction
  - **No helmet** — motorcycle rider detected without helmet class
  - **Triple riding** — 3+ persons on a single motorcycle
  - **Plate unreadable** — vehicle has no readable plate after N frames
  - **Overspeed** — velocity from homography exceeds speed limit
  - **Watchlist** — plate matches a watchlist entry
  - **Loitering** — person/vehicle dwells in a configured zone beyond threshold
  - **Abandoned object** — stationary object with no owner for N frames

### 5. Crowd analytics (CPU, partially async)
`_run_crowd_analytics(frame, frame_idx, person_bboxes)`:
- Zone density estimation via `ZoneDensityEstimator`
- Optical flow via `FlowAnalyzer` (Farneback, skipped every 4 frames to save CPU)
- Stampede risk scoring via `StampedeDetector`
- Fruin Level-of-Service classification
- Periodic crowd snapshots written to DB via `_crowd_executor`

### 6. Violence detection (GPU, async)
Two-stage system:
- **Stage 1 — Pose heuristics** (`PoseViolenceDetector`): YOLOv8-pose estimates body keypoints. Proximity IoU + wrist velocity heuristics flag potential fights. Runs every 3 frames.
- **Stage 2 — CLIP zero-shot** (`ViolenceClipClassifier`): CLIP (ViT-B/32) compares the frame against "violent fight" vs "cars on road / normal street scene" text prompts. Runs every 60 frames, or when Stage 1 fires.
- `IncidentManager` tracks incident lifecycle (open → confirmed → resolved).

### 7. Evidence persistence (fully async)
When a violation fires:
1. The violation event, raw frame, annotated frame (only drawn when there is a violation), and plate crop are handed off to `_db_executor`.
2. `_save_violation_bg()` runs in the background: encodes JPEGs, computes SHA-256, signs the payload with Ed25519, writes to SQLite via SQLAlchemy, pushes a WebSocket violation event to all connected officers.
3. The inference thread never waits for DB writes.

---

## Background Executor Pools

Each camera pipeline has 5 `ThreadPoolExecutor` instances:

| Executor | Workers | What runs in it |
|---|---|---|
| `_ocr_executor` | 1 | PaddleOCR PP-OCRv5 on plate crops |
| `_face_executor` | 2 | Face detection + ArcFace embedding |
| `_crowd_executor` | 1 | Crowd snapshot DB writes |
| `_db_executor` | 1 | Violation saves, plate sighting inserts/updates |
| `_reid_executor` | 2 | DINOv2-Small vehicle embedding |

All executors use `shutdown(wait=False, cancel_futures=True)` on camera stop so they don't block the main thread.

The result draining pattern is the same for all async executors:
1. Submit a `Future` at inference time, store in `_*_pending[track_id]`.
2. At the start of the **next** frame, drain completed futures into `_*_ready[track_id]`.
3. Apply the ready result during that frame's processing loop.
4. Never `future.result()` on the inference thread — that blocks.

---

## Live Video Feed — WebSocket Canvas Player

### Why Not MJPEG

The previous implementation used MJPEG: the browser loaded `<img src="/api/cameras/{cam}/mjpeg">` which received a `multipart/x-mixed-replace` HTTP response with JPEG frames pushed indefinitely. MJPEG has a fundamental flaw: **the browser buffers incoming multipart data**. When the server falls behind (inference thread slow, network hiccup), the browser accumulates a backlog of buffered frames and shows footage that is 5–10 seconds old. There is no way to skip to "latest" inside a multipart stream from the browser side.

### WebSocket Canvas Solution

**Server side (`src/api/server.py`):**

```python
@app.websocket("/ws/video/{camera_id}")
async def ws_video(websocket: WebSocket, camera_id: str):
    from src.pipeline.runner import get_live_frame
    await websocket.accept()
    last_bytes: Optional[bytes] = None
    try:
        while True:
            data = get_live_frame(camera_id)
            if data and data is not last_bytes:
                await websocket.send_bytes(data)
                last_bytes = data
            await asyncio.sleep(0.033)   # 30fps cap
    except WebSocketDisconnect:
        pass
```

`get_live_frame()` reads from `_live_frames[camera_id]` — a dict that the stream thread (ABOVE_NORMAL priority) updates at 30fps with the latest JPEG bytes. The WebSocket endpoint only sends when the bytes object reference changes (new frame), so it never resends the same frame twice.

**Browser side (`src/api/static/index.html`):**

```javascript
function _openFeedWS(cid) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/video/${cid}`);
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (e) => {
    const canvas = document.getElementById('feed-canvas-' + cid);
    const blob = new Blob([e.data], { type: 'image/jpeg' });
    createImageBitmap(blob).then(bmp => {
      const ctx = canvas.getContext('2d');
      if(canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas.width  = bmp.width;
        canvas.height = bmp.height;
      }
      ctx.drawImage(bmp, 0, 0);
      bmp.close();
    });
  };

  ws.onclose = () => {
    // Auto-reconnect after 3s if card still exists
    setTimeout(() => _openFeedWS(cid), 3000);
  };
}
```

Each camera feed is a `<canvas>` element. The WebSocket delivers raw JPEG bytes as `ArrayBuffer`. `createImageBitmap()` decodes the JPEG off the main thread (uses the browser's image decoder, hardware-accelerated). `ctx.drawImage(bmp, 0, 0)` paints it. `bmp.close()` immediately releases the bitmap memory.

**Why this is always current:** Each WebSocket message IS the latest frame — there is no queue, no buffer. If the browser's canvas draw is slow, the next message simply replaces the current one. The feed shows now, not 5 seconds ago.

**Resource management:**
- `_feedWS` map tracks all open WebSocket objects by camera ID
- `_closeFeedWS(cid)` nulls `onclose` before calling `close()` to suppress auto-reconnect on intentional closes
- IntersectionObserver closes WebSocket for cards scrolled out of viewport, reopens when they scroll back in (saves bandwidth for off-screen feeds)
- `visibilitychange` event closes all feeds when the browser tab is hidden, reopens when the tab is visible again

---

## Plate Recognition — How It Actually Works

Indian number plates follow the format: `KA 05 MJ 1234` (state code + district + letters + number). The system uses a two-engine consensus approach:

### Engine 1: fast-alpr (ONNX)
- Runs inline on the inference thread (no executor).
- Speed: ~5ms per frame on GPU.
- Detects the plate bounding box AND reads the text in one pass.
- Minimum confidence filter: 0.80 for generic format; UK plates (`^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$`) are rejected entirely — they appear as false positives on Indian footage.

### Engine 2: PaddleOCR PP-OCRv5
- Runs in `_ocr_executor` (background, never blocks inference).
- Speed: ~200-400ms on CPU (runs on CPU to leave GPU free for detection).
- Takes the plate crop from fast-alpr's detected bounding box.
- PP-OCRv5 is trained on multilingual text, works well on Indian plate fonts.

### PlateConsensusEngine
Both engines run on every frame where a plate is visible. The consensus engine collects all reads across all frames for a track and votes per character position:

```
Frame 1:  KA02NH7256  (fast-alpr 0.83)
Frame 2:  KA02NH7256  (paddleOCR)
Frame 3:  KA02NH725G  (fast-alpr 0.81)   ← last char misread
Frame 4:  KA02NH7256  (paddleOCR)
Frame 5:  KA02NH7256  (fast-alpr 0.85)
                 ↓
Consensus: KA02NH7256  (position 9: '6' wins 4-1)
```

After `min_reads_for_consensus=2` reads that produce the same string, the plate is accepted as a confirmed read. The UI **only shows a plate in the Plates tab once consensus is reached** — never partial/noisy single-frame reads.

Plate normalisation (`src/ocr/plate_normalize.py`) applies:
- Uppercase
- Remove spaces/dashes
- O/0, I/1 substitution rules specific to Indian plate typography
- Validate against Indian plate regex patterns

---

## Evidence Chain of Custody

Every violation record has:
1. **SHA-256** of the raw frame JPEG bytes
2. **Ed25519 signature** over `{code, camera_id, track_id, timestamp, frame_idx, plate_text, confidence, deployment_id}` concatenated with the frame hash
3. **Chain hash** — `SHA-256(prev_chain_hash + current_sha256)` — an append-only hash chain across all rows, so any deletion or modification of a past record is detectable
4. **`prev_id`** — foreign key to the previous violation record, forming an explicit linked list

The signing key is stored in `data/signing_key.pem` (Ed25519 private key). Verification uses the corresponding public key.

All evidence assets (raw frame, annotated frame, plate crop) are stored in the object store (`src/common/object_store.py`) which is either local filesystem (`data/evidence/`) or S3-compatible. Paths are stored in the violation row.

---

## Cross-Camera Re-Identification (ReID)

When the same vehicle appears on multiple cameras, SENTINEL links them under a single `global_id` (UUID).

### DINOv2-Small Embedder
`src/reid/embedder.py` extracts a 384-dimensional L2-normalised embedding from a 112×112 vehicle crop using DINOv2-Small (facebook/dinov2-small). DINOv2 was pretrained with self-supervised learning on diverse imagery — its features generalise to vehicles across lighting/angle changes.

### ReIDStore
`src/reid/store.py` maintains an in-memory FAISS (or numpy dot-product) index of all active vehicle embeddings, indexed by `(camera_id, track_id)`.

Search uses cosine similarity. Threshold: 0.85 (configurable). When a new embedding arrives:
1. Search for top-1 match from a **different camera** (same-camera matches are excluded — we already have track_id for that).
2. If similarity > threshold, merge under the matched `global_id`.
3. If a plate is known on the matched entry but not on the current track, propagate it back.

### Async Extraction Pattern
DINOv2 GPU inference is ~40-80ms. Running it inline would block the inference thread for every vehicle every 15 frames. Instead:
- `is_due(camera_id, track_id, frame_idx)` — checks if 15 frames have elapsed since last extraction
- `mark_extracting(...)` — reserves the slot (prevents double-queuing)
- Crop is submitted to `_reid_executor.submit(_reid_bg, ...)`
- Result arrives in `_reid_ready` on the next frame and is applied by `_drain_reid_results()`

---

## Database Schema (SQLite, SQLAlchemy ORM)

Key tables in `src/evidence/models.py`:

| Table | Purpose |
|---|---|
| `violations` | One row per traffic violation event. Has SHA-256, signature, chain hash. |
| `plate_sightings` | Every confirmed plate read on every camera. Updated (not re-inserted) as consensus improves. |
| `incidents` | Violence/crowd incidents with lifecycle (open → confirmed → resolved). |
| `face_captures` | Face crops from violence incidents, optionally matched to police records. |
| `camera_health` | Heartbeat + last violation timestamp per camera. |
| `reid_subjects` | Persisted ReID global IDs with vehicle attributes and cross-camera timestamps. |
| `crowd_snapshots` | Periodic crowd density/stampede snapshots. |
| `watchlist_entries` | Plate patterns that trigger immediate WS alert when seen. |
| `users` | Officers with email + bcrypt password + role (admin/supervisor/reviewer/viewer). |

All DB writes from the inference pipeline go through `_db_executor` so they never block video processing.

---

## API Authentication

JWT-based. `POST /auth/login` verifies bcrypt password and returns a signed JWT (HS256, 8-hour expiry). All state-changing endpoints require `Authorization: Bearer <token>`. Roles:
- `admin` — full access
- `supervisor` — approve/reject violations, manage watchlist, add cameras
- `reviewer` — approve/reject violations only
- `viewer` — read-only

Rate limiting is applied per-IP via `rate_limit` dependency.

---

## Configuration Files

| File | Purpose |
|---|---|
| `config/settings.py` (Pydantic `BaseSettings`) | All env-configurable settings: device, DB path, evidence dir, signing key path, JWT secret, bootstrap admin credentials |
| `config/cameras.yaml` | Per-camera specs: RTSP source, fps_cap, stop_line coordinates, signal_roi, loitering zones, crowd zones, homography points |
| `config/rules.yaml` | Per-rule enable/disable + thresholds. Loaded at startup; hot-reloaded via `POST /api/settings/rules/toggle` |

---

## Prometheus Metrics

`src/common/metrics.py` exports via `prometheus_client`:
- `frames_total{camera_id}` — counter of frames processed
- `violations_total{camera_id, code}` — counter per violation type per camera
- `review_action_total{action}` — approve/reject counts
- `review_backlog` — current pending violation count (gauge)

Scraped on `settings.prometheus_port` (default 9100).

---

## Key Design Decisions

### 1. One inference thread per camera, not one GPU thread for all cameras
Each camera's `_inference_loop` runs independently at its own rate (10fps cap). This means 5 cameras × 10fps = 50 inference calls per second max, but they are all serialised on the GPU (CUDA serialises kernel dispatch from multiple CPU threads). The benefit: each camera's pipeline state (tracker, rules engine, plate consensus, loitering state) stays on one thread with no locking required.

### 2. Frame rate separation: display at 30fps, inference at 10fps
The stream thread updates the live frame store at 30fps (smooth video). The inference thread runs at 10fps (enough to catch violations, saves GPU budget). These are completely independent — slowing down inference doesn't affect the live display at all.

### 3. Async everything that touches disk or takes >10ms
Before async executors, a single `store.save()` call took 20-31 seconds because SQLite under Windows I/O scheduler would sleep the calling thread. The inference thread would freeze for that entire duration, dropping all frames from all cameras. Every non-GPU operation that could be slow is now in an executor.

### 4. BELOW_NORMAL priority on inference thread
Even with async executors, the inference thread competes with executor threads (OCR, ReID, DB) for CPU timeslices. Under Windows, when executor threads ran at NORMAL priority and inference ran at NORMAL priority, the OS would occasionally preempt the inference thread for 7-9 seconds while executor threads ran. The slow log showed `ann=7875ms` — that was `time.monotonic()` capturing 7 seconds of OS suspension, not actual work. Setting inference to BELOW_NORMAL and stream/buffer to ABOVE_NORMAL resolved this entirely.

### 5. Annotate only on violations, not every frame
Previously `annotate(frame, ...)` ran on every frame to draw bounding boxes on the live feed. `annotate()` takes ~50ms because it calls OpenCV drawing functions for every detected object. This was the source of steady 50ms overhead per frame even when nothing was happening. Now annotation only runs when a violation event fires — the annotated frame is the evidence image, not the live feed. The live feed shows raw unmodified frames.
