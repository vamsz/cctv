"""Fine-tune YOLO11n on data/plate_ft/ for license plate detection.

Why train our own detector when fast-alpr already has one:
  fast-alpr's plate detector (yolo-v9-t-384) was trained on a global
  plate corpus dominated by European / North American formats. On
  Indian CCTV footage it under-detects oblique-angle plates and over-
  crops to the visible glyphs (which our OCR pipeline then has to
  pad back out). Training on data/plate_ft/ — 6675 labelled Indian
  plate images — gives us a detector tuned for our actual deployment.

Usage:
  python scripts/train_plate_detector.py
  # → models/plate_ft.pt (~6 MB)

Training takes ~30 min on a 3070 Ti at 50 epochs. The result is
loaded at runtime by Detector when models/plate_ft.pt exists.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_plate")

DATASET_YAML = ROOT / "data" / "plate_ft" / "data.yaml"
OUTPUT_PATH = ROOT / "models" / "plate_ft.pt"


def main() -> None:
    if not DATASET_YAML.exists():
        log.error("dataset yaml missing: %s", DATASET_YAML)
        sys.exit(1)

    from ultralytics import YOLO

    log.info("Training YOLO11n plate detector on %s", DATASET_YAML)
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,                  # CUDA 0
        patience=10,
        name="plate_ft",
        project=str(ROOT / "runs"),
        # Augmentations specific to the plate-recognition use case:
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=8.0,                # plates are NEVER upside down
        translate=0.10,
        scale=0.5,                  # important — plates appear at many distances
        mosaic=0.8,
        mixup=0.0,                  # mixup hurts text-shape models
        flipud=0.0,                 # plates are not vertically symmetric
        fliplr=0.0,                 # plates are LEFT-TO-RIGHT — never flip
        cache=True,                 # 6675 images at 640 fits in RAM
        verbose=True,
    )

    # Locate the best.pt produced by ultralytics and copy to models/
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        log.error("training finished but best.pt not found at %s", best)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(best, OUTPUT_PATH)
    log.info("Saved fine-tuned plate detector to %s", OUTPUT_PATH)
    log.info(
        "Detector::__init__ auto-loads this when use_fast_alpr=False OR when "
        "plate_ft.pt is present. To wire it in, set settings.use_fast_alpr=False "
        "OR rename to models/plate.pt.",
    )


if __name__ == "__main__":
    main()
