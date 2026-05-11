"""Extract plate crops from Kaggle dataset for hand-labelling.

Workflow:
  1. Reads YOLO bounding box annotations from kaggle_ocr/labels/
  2. Crops the license plate from kaggle_ocr/images/
  3. Uses fine-tuned OCR model to predict the text
  4. Saves crop to kaggle_ocr/extracted/ as <image_name>_crop<idx>_<ocr_guess>.jpg
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Suppress noisy boot messages
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_kaggle")

KAGGLE_DIR = ROOT / "kaggle_ocr"
IMAGES_DIR = KAGGLE_DIR / "images"
LABELS_DIR = KAGGLE_DIR / "labels"
OUT_DIR = KAGGLE_DIR / "extracted"


def safe_filename(text: str) -> str:
    """Make a filesystem-safe filename from an OCR string."""
    s = "".join(c for c in text.upper() if c.isalnum())
    return s if s else "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=0.25,
                    help="padding ratio around the bounding box (default 0.25)")
    args = ap.parse_args()

    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        log.error("Kaggle dataset directories not found: %s or %s", IMAGES_DIR, LABELS_DIR)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather images
    images = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        images.extend(IMAGES_DIR.glob(ext))
    images = sorted(images)

    if not images:
        log.error("No images found in %s", IMAGES_DIR)
        sys.exit(1)

    log.info("Found %d images in Kaggle dataset. Loading OCR engine...", len(images))

    from src.pipeline.runner import _crop_padded
    
    try:
        from fast_plate_ocr import LicensePlateRecognizer as _Rec
    except ImportError:
        from fast_plate_ocr import ONNXPlateRecognizer as _Rec
        
    log.info("Loading fast-plate-ocr...")
    ocr = _Rec(hub_ocr_model="global-plates-mobile-vit-v2-model")
    log.info("Loaded fast-plate-ocr. Writing extracted crops to %s", OUT_DIR)

    total_saved = 0
    missing_labels = 0
    t0 = time.time()

    for img_path in images:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning("Could not read image: %s", img_path.name)
            continue

        img_h, img_w = frame.shape[:2]

        crop_idx = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Parse YOLO format: class cx cy w h (normalized 0-1)
            cx, cy, w, h = map(float, parts[1:5])
            
            # Convert to absolute pixel coordinates
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            abs_w = w * img_w
            abs_h = h * img_h
            
            x1 = int(abs_cx - abs_w / 2)
            y1 = int(abs_cy - abs_h / 2)
            x2 = int(abs_cx + abs_w / 2)
            y2 = int(abs_cy + abs_h / 2)
            
            xyxy = [x1, y1, x2, y2]
            
            # Crop with padding
            crop = _crop_padded(frame, xyxy, pad=args.pad)
            if crop is None or crop.size == 0:
                continue

            # Get OCR prediction via fast-plate-ocr
            try:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # fast-plate-ocr expects (H, W, 1) or (H, W) depending on version, try grayscale first
                result = ocr.run(gray_crop, return_confidence=True)
                if result:
                    item = result[0]
                    if hasattr(item, "plate"):
                        guess = str(item.plate).strip().upper()
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        guess = str(item[0]).strip().upper()
                    else:
                        guess = str(item).strip().upper()
                    
                    guess = "".join(c for c in guess if c.isalnum())
                    if not guess:
                        guess = "UNREAD"
                else:
                    guess = "UNREAD"
            except Exception as e:
                guess = "UNREAD"
                
            fname_base = safe_filename(guess)

            # Save the file
            out_name = f"{img_path.stem}_crop{crop_idx}_{fname_base}.jpg"
            cv2.imwrite(str(OUT_DIR / out_name), crop)
            
            total_saved += 1
            crop_idx += 1
            
            if total_saved <= 5 or total_saved % 100 == 0:
                log.info("Saved crop %d: %s (from %s)", total_saved, out_name, img_path.name)

    log.info("=" * 70)
    log.info("Extracted %d total plate crops to %s in %.1fs", total_saved, OUT_DIR, time.time() - t0)
    if missing_labels > 0:
        log.warning("Skipped %d images that did not have matching label .txt files.", missing_labels)
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Open kaggle_ocr/extracted/ in your file browser")
    log.info("  2. Review the images and RENAME them if the OCR guess is wrong")
    log.info("  3. Train the OCR: python scripts/train_plate_recognizer.py")


if __name__ == "__main__":
    main()
