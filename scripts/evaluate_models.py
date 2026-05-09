"""Evaluate fine-tuned helmet and plate models on their validation sets.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --model helmet   # helmet only
    python scripts/evaluate_models.py --model plate    # plate only
    python scripts/evaluate_models.py --conf 0.30      # custom conf threshold

Reports per-class precision / recall / mAP50 / mAP50-95 and compares
the fine-tuned model against the base model if both are present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def eval_model(name: str, weights: Path, data_yaml: Path, device: int = 0, conf: float = 0.25) -> None:
    from ultralytics import YOLO
    print(f"\n{'='*60}")
    print(f"  {name}: {weights.name}")
    print(f"  data: {data_yaml}")
    print(f"{'='*60}")
    if not weights.exists():
        print(f"  [SKIP] weights not found: {weights}")
        return
    if not data_yaml.exists():
        print(f"  [SKIP] data.yaml not found: {data_yaml}")
        return
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), device=device, conf=conf, verbose=True)
    box = metrics.box
    print(f"\n  mAP50      : {box.map50:.4f}")
    print(f"  mAP50-95   : {box.map:.4f}")
    print(f"  Precision  : {box.mp:.4f}")
    print(f"  Recall     : {box.mr:.4f}")
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        names = model.names
        print("\n  Per-class mAP50:")
        for i, cls_idx in enumerate(box.ap_class_index):
            print(f"    {names[int(cls_idx)]:20s} {box.ap50[i]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["helmet", "plate", "all"], default="all")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    # Prefer fine-tuned models when available
    models_dir = ROOT / "models"
    data_dir   = ROOT / "data"

    if args.model in ("helmet", "all"):
        weights = models_dir / "helmet_ft.pt"
        if not weights.exists():
            weights = models_dir / "helmet.pt"
        eval_model(
            "Helmet model",
            weights,
            data_dir / "helmet_combined" / "data.yaml",
            args.device, args.conf,
        )

    if args.model in ("plate", "all"):
        weights = models_dir / "plate_ft.pt"
        if not weights.exists():
            weights = models_dir / "plate.pt"
        eval_model(
            "Plate model",
            weights,
            data_dir / "plate_ft" / "data.yaml",
            args.device, args.conf,
        )


if __name__ == "__main__":
    main()
