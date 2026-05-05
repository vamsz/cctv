# CCTV Enforcement — Quickstart (legacy — see HOW_TO_RUN.md for the full guide)

## What's working right now

- **Detection**: YOLO11 + helmet detector + plate detector (all loaded from HuggingFace)
- **Tracking**: ByteTrack via supervision
- **OCR**: EasyOCR with 6-variant preprocessing + Indian plate format normalization
- **Database**: SQLite (auto-created), evidence stored with SHA-256 + Ed25519 chain
- **API**: FastAPI on port 8000, no auth (dev mode)
- **Dashboard**: Card-based review UI at `/`

## One-time setup

```bash
pip install -r requirements.txt
python scripts/download_models.py
```

This pulls 3 model files into `./models/`:
- `yolo11n.pt` (5.4 MB) — vehicles/persons
- `plate.pt` (38.6 MB) — license plate detector
- `helmet.pt` (18.3 MB) — helmet/no-helmet detector

## Test the OCR

```bash
python scripts/test_ocr.py --synthetic
```

Renders 5 synthetic Indian plates and runs them through the full OCR pipeline.
Expect ~3-5 of 5 to be read correctly (font-induced confusions like S↔5 are real OCR errors).

## Test on a video

```bash
python scripts/test_video.py --video path/to/clip.mp4 --device cpu --no-display --save data/output.mp4
```

Flags:
- `--device cpu` for systems without CUDA
- `--every-n 3` skip frames for faster CPU testing
- `--debug-crops` save every plate crop to `data/debug_crops/`
- `--max-frames 30` stop after N frames

**Important**: The plates in your source video need to be at least ~80px wide
in the source frame for OCR to work. Low-resolution stock footage won't read.

## See the dashboard

```bash
python scripts/seed_demo.py     # Seed 8 sample violations
python scripts/run_api.py       # Start API on :8000
```

Open http://localhost:8000/ — review dashboard with approve/reject buttons,
filters by status/code/camera/plate.

## Run the full real-time pipeline

Edit `config/cameras.yaml` to point at your RTSP/file source:

```yaml
cameras:
  - id: CAM01
    source: "rtsp://user:pass@host:554/stream"
    fps_cap: 15
    stop_line: [[100, 600], [1820, 600]]
    signal_roi: [1500, 80, 1620, 280]
    direction: [0.0, -1.0]
    enabled: true
```

Calibrate stop-line and signal ROI:

```bash
python scripts/calibrate.py --source rtsp://... --id CAM01
```

Then start the pipeline + API:

```bash
python scripts/run_pipeline.py    # detector + rules + DB writes
python scripts/run_api.py         # dashboard
```

## Troubleshooting

**OCR returns nothing on a video** → plates are too small in source frame. Need 80+px width.

**Pipeline is slow on CPU** → use `--every-n 3` and `--device cpu` flags. CPU inference is 5-10x slower than GPU. Plan: NVIDIA GPU for production.

**Dashboard shows no images** → make sure API server is running and `data/evidence/` exists.

**Models not found** → re-run `python scripts/download_models.py`.
