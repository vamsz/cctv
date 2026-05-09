"""Extract license_plate annotations from helmet_combined and build a plate-only dataset.

The helmet_combined dataset has 3 classes:
  0 = helmet, 1 = no_helmet, 2 = license_plate (from Roboflow split only)

This script filters to class 2, remaps it to class 0, and produces:
  data/plate_ft/
    train/images/, train/labels/
    val/images/,   val/labels/
    data.yaml

Usage:
    python scripts/prepare_plate_dataset.py
Then:
    python scripts/train_plate_yolo.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
SRC      = ROOT / "data" / "helmet_combined"
OUT      = ROOT / "data" / "plate_ft"
PLATE_ID = 2   # class index in helmet_combined for license_plate


def extract_split(split: str) -> int:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"
    dst_img = OUT / split / "images"
    dst_lbl = OUT / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for lbl_path in src_lbl.glob("*.txt"):
        lines = lbl_path.read_text().splitlines()
        plate_lines = [
            "0 " + " ".join(parts[1:])
            for line in lines
            if line.strip() and (parts := line.split()) and int(parts[0]) == PLATE_ID
        ]
        if not plate_lines:
            continue  # image has no plate annotations — skip

        # Copy image
        copied = False
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            src = src_img / (lbl_path.stem + ext)
            if src.exists():
                shutil.copy2(src, dst_img / src.name)
                copied = True
                break
        if not copied:
            continue

        (dst_lbl / lbl_path.name).write_text("\n".join(plate_lines))
        count += 1

    print(f"  {split}: {count} images with plate annotations")
    return count


if __name__ == "__main__":
    if not SRC.exists():
        raise SystemExit(f"Source dataset not found: {SRC}\nRun scripts/prepare_helmet_dataset.py first")

    if OUT.exists():
        shutil.rmtree(OUT)

    print("=== Extracting plate annotations from helmet_combined ===")
    n_train = extract_split("train")
    n_val   = extract_split("val")

    yaml = f"""path: {OUT.as_posix()}
train: train/images
val: val/images

nc: 1
names: ["license_plate"]
"""
    (OUT / "data.yaml").write_text(yaml)
    print(f"\nWrote data.yaml  —  {n_train} train / {n_val} val samples")
    print(f"Dataset ready at: {OUT}")
    print("\nNext step:  python scripts/train_plate_yolo.py")
