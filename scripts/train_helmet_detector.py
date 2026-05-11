"""Fine-tune YOLO11n on data/helmet_combined/ for helmet detection.

Dataset: 8757 train + 482 val images, 3 classes (helmet, no_helmet,
license_plate). The license_plate class is ignored at inference but
having it in training acts as a useful negative for the helmet head —
it stops the model from confusing low-resolution plates with helmets.

Usage:
  python scripts/train_helmet_detector.py
  # → models/helmet_ft.pt   (~6 MB, picked up automatically by Detector)

Training takes ~45 min on a 3070 Ti at 80 epochs. Detector auto-loads
models/helmet_ft.pt over models/helmet.pt when both exist.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_helmet")

DATASET_YAML = ROOT / "data" / "helmet_combined" / "data.yaml"
OUTPUT_PATH = ROOT / "models" / "helmet_ft.pt"


def main() -> None:
    if not DATASET_YAML.exists():
        log.error("dataset yaml missing: %s", DATASET_YAML)
        sys.exit(1)

    from ultralytics import YOLO

    os.environ["PYTHONWARNINGS"] = "ignore"  # Silence urllib3 warnings in dataloader subprocesses

    log.info("Training YOLO11n helmet detector on %s", DATASET_YAML)
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,                # CUDA 0
        patience=15,
        name="helmet_ft",
        project=str(ROOT / "runs"),
        # Aug params tuned for two-wheeler rider footage:
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=10.0,            # mild — heads aren't usually tilted past 30°
        translate=0.10,
        scale=0.5,               # important — riders appear at varying distances
        mosaic=1.0,               # heavy mosaic helps small-object detection
        mixup=0.1,                # mixup is fine for object detection
        flipud=0.0,               # never flip vertically
        fliplr=0.5,               # horizontal flip is symmetric for helmets
        cache="disk",             # 'disk' avoids RAM OOMs and is much faster than raw reads
        workers=4,                # Reduced from 8 to prevent CPU context-switch overhead with disk cache
        cls=0.7,                  # weight classification loss higher than usual
                                  # (helmet vs no_helmet is the whole point)
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        log.error("training finished but best.pt not found at %s", best)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(best, OUTPUT_PATH)
    log.info("Saved fine-tuned helmet detector to %s", OUTPUT_PATH)
    log.info(
        "Detector::__init__ already auto-upgrades to helmet_ft.pt when present. "
        "Re-run reset_run.ps1 and the new weights will be used immediately. "
        "Don't forget to set helmet.enabled: true in config/rules.yaml.",
    )


if __name__ == "__main__":
    main()
