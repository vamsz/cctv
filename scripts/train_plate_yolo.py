"""Fine-tune YOLO11n for Indian license-plate detection on plate_ft dataset.

Usage:
    python scripts/prepare_plate_dataset.py   # first, extract annotations
    python scripts/train_plate_yolo.py        # then train

Output:
    models/plate_ft.pt     ← best checkpoint
    runs/plate_ft/         ← full training artifacts
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "plate_ft" / "data.yaml"
BASE    = ROOT / "models" / "yolo11n.pt"
OUT_DIR = ROOT / "runs" / "plate_ft"
FINAL   = ROOT / "models" / "plate_ft.pt"


def main() -> None:
    from ultralytics import YOLO

    assert DATA.exists(), f"Dataset not found: {DATA}  →  run scripts/prepare_plate_dataset.py first"
    assert BASE.exists(), f"Base weights not found: {BASE}"

    print(f"Base      : {BASE}")
    print(f"Dataset   : {DATA}")
    print(f"Output    : {OUT_DIR}")

    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        epochs=80,
        patience=15,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(ROOT / "runs"),
        name="plate_ft",
        exist_ok=True,
        # Augmentation — plates benefit from aggressive scale/rotation
        augment=True,
        mosaic=0.5,
        mixup=0.0,
        degrees=5.0,
        scale=0.6,
        fliplr=0.0,     # plates are directional — don't flip horizontally
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
    )

    best = OUT_DIR / "weights" / "best.pt"
    if not best.exists():
        candidates = list((ROOT / "runs").rglob("plate_ft*/weights/best.pt"))
        if candidates:
            best = candidates[-1]

    if best.exists():
        shutil.copy2(best, FINAL)
        print(f"\n✓ Best weights → {FINAL}")
    else:
        print(f"\n⚠  best.pt not found — check {OUT_DIR}")


if __name__ == "__main__":
    main()
