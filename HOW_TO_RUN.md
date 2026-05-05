# PRAHARI — How to Run

## Requirements

| Item | Version / Note |
|---|---|
| Python | 3.11+ |
| CUDA | 12.x (RTX 3070 Ti tested) — CPU fallback automatic |
| GPU VRAM | 4 GB minimum, 8 GB recommended (3-camera setup) |
| Docker | 24+ with nvidia-container-toolkit (for containerised run) |

---

## Option A: Local (no Docker)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download model weights
```bash
python scripts/download_models.py
```
Downloads `yolo11n.pt`, `helmet.pt`, `plate.pt` into `./models/`.
`yolo11n-pose.pt` auto-downloads on first pose inference via ultralytics cache.

### 3. Configure cameras
Edit `config/cameras.yaml`:
```yaml
cameras:
  - id: CAM01
    source: "rtsp://user:pass@192.168.1.10:554/stream"  # or path/to/file.mp4
    fps_cap: 15
    stop_line: [[100, 600], [1820, 600]]  # calibrate with scripts/calibrate.py
    signal_roi: [1500, 80, 1620, 280]
    direction: [0.0, -1.0]
    enabled: true
    loitering_zones:
      - id: lz1
        label: "Gate Area"
        polygon: [[0,400],[640,400],[640,720],[0,720]]
        threshold_seconds: 120
    crowd_zones:
      - id: cz1
        label: "Full Frame"
        polygon: [[0,0],[1280,0],[1280,720],[0,720]]
```

### 4. Calibrate stop-line (one-time per camera)
```bash
python scripts/calibrate.py --source rtsp://... --id CAM01
```
Click the stop-line endpoints. Press S to save. Saves coordinates to cameras.yaml.

### 5. Seed demo data (optional — for UI testing)
```bash
python scripts/seed_demo.py
```

### 6. Start
```bash
# Terminal 1 — inference pipeline
python scripts/run_pipeline.py

# Terminal 2 — API + dashboard
python scripts/run_api.py
```

Open **http://localhost:8000** in browser. Sign in: `admin@local` / `admin`.

---

## Option B: Docker Compose (recommended for deployment)

```bash
# 1. Copy env template and fill in secrets
cp .env.example .env

# 2. Build and start
docker compose up --build -d

# 3. Open dashboard
open http://localhost:8000    # enforcement console
open http://localhost:3000    # grafana (admin/admin first time)
```

GPU access requires `nvidia-container-toolkit` installed on the host and Docker configured for GPU.

---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| **Violations** | All traffic violations with approve/reject workflow |
| **Incidents** | Violence / loitering / abandoned object events |
| **Crowd** | Live crowd density + stampede risk per camera |
| **Live Feeds** | MJPEG streams from all cameras (pipeline must be running) |
| **Analytics** | Charts — violations by type, camera, review status |
| **Watchlist** | Watchlisted plate patterns |
| **Subjects** | Cross-camera vehicle identities (ReID) |
| **Settings** | Enable/disable rules at runtime |

**Keyboard shortcuts** (Violations tab):
- `↑ / ↓` — navigate rows
- `A` — approve selected violation
- `X` — reject selected violation
- `Esc` — collapse expanded row
- `F5` — reload violations

---

## India-Tuned Thresholds (config/rules.yaml)

All defaults are calibrated for Indian conditions:

| Setting | Value | Reason |
|---|---|---|
| Crowd warning density | 4.0 p/m² | Normal markets run 2-4 p/m² |
| Crowd danger density | 7.0 p/m² | Festival crush is 7+ (5 too sensitive) |
| Stampede divergence min | 0.35 | Bidirectional roads have high baseline |
| Stampede counter-flow min | 0.30 | Normal Indian traffic is bidirectional |
| Violence wrist velocity | 12 px/frame | Gesturing during arguments is normal |
| Loitering threshold | 120 s | Vendors/commuters sit long |
| Abandoned object min | 300 frames | Street clutter is high |

To re-tune for a specific deployment, edit `config/rules.yaml`. Rule enable/disable takes effect immediately via the Settings tab. Threshold changes require a pipeline restart.

---

## TensorRT (max throughput on RTX 3070 Ti)

Build TRT engines **on the deployment machine** (engines are GPU-specific):

```bash
python scripts/export_tensorrt.py --weights models/yolo11n.pt --imgsz 640 --half
python scripts/export_tensorrt.py --weights models/helmet.pt   --imgsz 640 --half
python scripts/export_tensorrt.py --weights models/plate.pt    --imgsz 640 --half
```

Then update `.env`:
```env
DETECTOR_WEIGHTS=models/yolo11n.engine
HELMET_WEIGHTS=models/helmet.engine
PLATE_WEIGHTS=models/plate.engine
```

Expected throughput on RTX 3070 Ti:
- FP32 PyTorch: ~8-10 cameras @ 15fps
- FP16 TensorRT: ~20-25 cameras @ 15fps

---

## Benchmarking

```bash
# How many concurrent RTSP streams can the GPU handle?
python scripts/benchmark_streams.py --device cuda:0 --max-cameras 10
```

---

## Reset / Reseed

```bash
python scripts/reset_data.py --yes          # wipe all data
python scripts/seed_demo.py                 # add 52 demo violations
python scripts/reset_data.py --seed-only --yes  # remove only seed data
```

---

## Monitoring (Grafana)

Prometheus scrapes pipeline metrics at `host:9090` and API metrics at `host:9091`.
Key metrics:
- `cctv_frames_processed_total` — fps per camera
- `cctv_frame_drops_total` — skipped frames (inference slower than capture)
- `cctv_inference_latency_seconds` — p50/p95/p99 per camera
- `cctv_violations_total` — violations fired by type
- `cctv_review_backlog` — pending operator reviews
- `cctv_camera_up` — 1 if live, 0 if offline
