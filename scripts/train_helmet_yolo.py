"""Fine-tune YOLO11n on the merged helmet+plate dataset.

Usage:
    python scripts/train_helmet_yolo.py

Outputs:
    models/helmet_ft.pt        ← best checkpoint, ready to drop into pipeline
    runs/helmet_ft/            ← full training artifacts
"""
from pathlib import Path
import shutil

ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data" / "helmet_combined" / "data.yaml"
BASE     = ROOT / "models" / "yolo11n.pt"
OUT_DIR  = ROOT / "runs" / "helmet_ft"
FINAL    = ROOT / "models" / "helmet_ft.pt"


def main():
    from ultralytics import YOLO

    assert DATA.exists(), f"Dataset not found: {DATA}  →  run scripts/prepare_helmet_dataset.py first"
    assert BASE.exists(), f"Base weights not found: {BASE}"

    print(f"Base weights : {BASE}")
    print(f"Dataset      : {DATA}")
    print(f"Output dir   : {OUT_DIR}")

    model = YOLO(str(BASE))

    results = model.train(
        data=str(DATA),
        epochs=100,
        patience=20,          # early stop if no improvement for 20 epochs
        imgsz=640,
        batch=16,             # safe for RTX 3070 Ti 8.6 GB at 640px
        device=0,             # GPU
        workers=4,
        project=str(ROOT / "runs"),
        name="helmet_ft",
        exist_ok=True,
        # Augmentation — aggressive for small helmet objects
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Save
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
    )

    # Copy best weights to models/
    best = OUT_DIR / "weights" / "best.pt"
    if not best.exists():
        best = Path(str(OUT_DIR) + "2") / "weights" / "best.pt"  # ultralytics increments
    if best.exists():
        shutil.copy2(best, FINAL)
        print(f"\n✓ Best weights saved to {FINAL}")
    else:
        # Search runs/ for best.pt
        candidates = list((ROOT / "runs").rglob("helmet_ft*/weights/best.pt"))
        if candidates:
            shutil.copy2(candidates[-1], FINAL)
            print(f"\n✓ Best weights saved to {FINAL}")
        else:
            print(f"\n⚠ Could not find best.pt — check {OUT_DIR}")

    print("\nValidation metrics:")
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
