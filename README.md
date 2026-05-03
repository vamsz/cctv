# CCTV Traffic Enforcement Pipeline

Production-shaped enforcement system for Indian traffic CCTV. Detects
violations, captures audited evidence, and exposes a review console for
operators to approve or reject each event before it becomes a challan.

The system is intentionally a **multi-stage pipeline**, not a single
"AI model." Each stage is independently swappable and auditable — which
matters when a violation is contested.

```
   RTSP/CCTV ──▶ Ingest ──▶ Detection ──▶ Tracking ──▶ Plate OCR ──▶ Rules ──▶ Evidence ──▶ Review
                (OpenCV)   (YOLO11 +    (ByteTrack    (PaddleOCR    (helmet,    (SQLite +    (FastAPI
                            helmet,      via           + IN-plate    red-light,  filesystem)  dashboard)
                            plate)       supervision)  normalize)    plate-unread)
```

## Why this architecture

- **Multi-model pipeline**, not one black box. Each stage produces
  evidence the next stage uses; a court can be shown the helmet crop,
  the OCR confidence, and the stop-line crossing event independently.
- **Tracker-gated rules.** A violation only fires after the *same*
  vehicle/rider exhibits it across multiple frames. No single-frame
  hallucinations end up in the issuance queue.
- **Human-in-the-loop by default.** Every event lands in the review
  console as `pending`. Nothing auto-issues.
- **Indian-first.** License plate normalization knows the state-code
  grammar (old format and Bharat series), and applies position-aware
  OCR character corrections.

## Repository layout

```
config/
  cameras.yaml          per-camera RTSP, stop line, signal ROI, lawful direction
  rules.yaml            tunable thresholds for every rule
  settings.py           pydantic Settings; reads .env
src/
  ingest/               threaded RTSP/file reader, always-latest-frame
  detection/            YOLO wrapper fusing general + helmet + plate heads
  tracking/             ByteTrack via supervision, vehicle/person identity
  ocr/                  PaddleOCR + Indian-plate format normalization
  signal/               traffic-light state classifier (HSV; CNN-replaceable)
  rules/                helmet, red-light, plate-unreadable; cooldowns; geometry
  evidence/             SQLAlchemy ORM + filesystem layout
  pipeline/             per-camera orchestrator (ingest -> rules -> persist)
  api/                  FastAPI + static HTML/JS review console
scripts/
  run_pipeline.py       launch the inference pipeline
  run_api.py            launch the review API + dashboard
  calibrate.py          interactive stop-line / signal-ROI picker
  download_models.py    fetch base YOLO11n
  train_helmet.py       fine-tune helmet/no_helmet head
  train_plate.py        fine-tune plate detection head
  export_tensorrt.py    export .pt -> ONNX -> TensorRT engine
Dockerfile
docker-compose.yml
requirements.txt
.env.example
```

## Quickstart (local, single camera, sample video)

1. **Install Python deps** (Python 3.11 recommended):

   ```bash
   pip install -r requirements.txt
   ```

   PaddleOCR will pull additional model files on first run. On a CPU-only
   machine, replace `paddlepaddle` with `paddlepaddle` CPU build (already
   the default in `requirements.txt`).

2. **Download base weights:**

   ```bash
   python scripts/download_models.py
   ```

3. **Drop a sample video** at `./data/samples/traffic.mp4`, or edit
   `config/cameras.yaml` to point `SAMPLE` at any file you have.
   Enable the `SAMPLE` camera by setting `enabled: true`.

4. **Calibrate the stop line and signal ROI** for your camera:

   ```bash
   python scripts/calibrate.py --source ./data/samples/traffic.mp4 --id SAMPLE
   ```

   - Left-click two points → stop line
   - Right-drag → signal-light ROI rectangle
   - Arrow keys → rotate the lawful-direction arrow until it points
     where vehicles travel after they cross the stop line
   - Press `s` → prints a YAML snippet to paste into `config/cameras.yaml`

5. **Copy `.env.example` to `.env`** and adjust `DEVICE` (`cpu` if no GPU).

6. **Run the pipeline:**

   ```bash
   python scripts/run_pipeline.py
   ```

7. **In another terminal, run the review console:**

   ```bash
   python scripts/run_api.py
   ```

   Open <http://localhost:8000>. Pending violations appear as cards;
   click **Approve** or **Reject**.

## Production deployment (Docker, GPU)

Requirements: NVIDIA GPU with driver ≥ 535, `nvidia-container-toolkit`
installed on the host.

```bash
cp .env.example .env
# edit .env — set DEVICE=cuda:0, real DETECTOR_WEIGHTS path, etc.
docker compose build
docker compose up -d
```

Two containers:
- `cctv-pipeline` — runs inference, writes evidence
- `cctv-api`      — serves dashboard at port 8000

Both share the same `./data` and `./config` volumes.

## Training your own heads (the part that makes it actually work in India)

The base YOLO11n is COCO-trained — it knows "motorcycle" and "person"
but not "helmet/no_helmet" or "Indian plate." You **must** train these
on labeled Indian-traffic data to get acceptable accuracy.

### Helmet / no-helmet

Annotate the head region of each rider with two classes: `helmet` and
`no_helmet`. ~3k–5k images split 90/10 train/val is the practical
minimum; aim for 15k for production.

```bash
python scripts/train_helmet.py --data datasets/helmet/data.yaml --epochs 80
cp runs/detect/helmet/weights/best.pt models/helmet.pt
```

### License plate detection

Annotate the plate rectangle (one class). Same scale as above.

```bash
python scripts/train_plate.py --data datasets/plate/data.yaml --epochs 80
cp runs/detect/plate/weights/best.pt models/plate.pt
```

### Plate OCR fine-tuning (optional, recommended for production)

PaddleOCR ships with a strong English recognizer that handles Indian
plates surprisingly well after the position-aware normalization in
`src/ocr/plate_normalize.py`. For higher accuracy on faded/old plates,
fine-tune the recognizer on a labeled plate-text dataset using the
official PaddleOCR training pipeline. The pipeline doesn't care which
recognizer is loaded — it just calls `PlateOCR.read()`.

### Engine compilation for inference speed

After training, export to TensorRT for ~3–5× throughput on the same GPU:

```bash
python scripts/export_tensorrt.py --weights models/helmet.pt --imgsz 640 --half
# yields models/helmet.engine
```

Update `.env` to point `HELMET_WEIGHTS` (and the others) at the `.engine`
file. Ultralytics' inference path automatically uses TensorRT when given
an `.engine`.

## What's intentionally not in v1

- **Wrong-way driving** — needs lane geometry per camera; add as a new
  rule in `src/rules/`.
- **Triple riding** — needs a person count per two-wheeler; straightforward
  extension once the helmet head is solid.
- **Speed estimation** — needs camera calibration (homography); add a
  module under `src/speed/`.
- **Chalana app integration** — explicitly deferred per spec. The
  evidence schema is already shaped for the handoff: each approved row
  has `plate_text`, `code`, `timestamp`, `camera_id`, and image asset
  paths — push it as JSON to whatever issuance API you're given.

## Operational notes

- **One process per camera is fine up to ~6 cameras on a single GPU.**
  Above that, run multiple `cctv-pipeline` containers, one per group of
  cameras, sharing the same evidence DB.
- **SQLite** is fine for one site. For city-wide, swap `DATABASE_URL` to
  `postgresql://...`. The schema is identical; SQLAlchemy handles both.
- **Cooldowns** in `config/rules.yaml` prevent the same vehicle from
  generating multiple events for the same offense — tune per
  intersection.
- **Evidence retention.** Plan a cron/sweep job to archive evidence
  older than N days off the inference host. Not included in v1.

## License & responsibility

This system produces **evidence**, not verdicts. A human operator must
approve every event before it becomes a challan. The author of this
repo accepts no liability for downstream issuance decisions.
