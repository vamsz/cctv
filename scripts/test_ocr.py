"""Standalone OCR sanity check — run on a single image or synthetic plate.

Usage:
    python scripts/test_ocr.py --image path/to/plate.jpg
    python scripts/test_ocr.py --synthetic
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr.plate_ocr import PlateOCR
from src.ocr.plate_normalize import normalize_indian_plate


def make_synthetic_plate(text: str = "TS09EA1234") -> np.ndarray:
    """Generate a clean, large, high-contrast Indian-style plate image."""
    W, H = 600, 150
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (W - 5, H - 5), (0, 0, 0), 4)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 2.5
    thick = 5
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (W - tw) // 2
    y = (H + th) // 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to plate crop image")
    ap.add_argument("--synthetic", action="store_true", help="Test on synthetic plates")
    ap.add_argument("--save", default=None, help="Save synthetic plate to this path")
    args = ap.parse_args()

    print("Loading EasyOCR...")
    ocr = PlateOCR(lang="en", use_gpu=False, fast_mode=False)
    print("OCR ready.\n")

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read {args.image}")
            sys.exit(1)
        print(f"Image: {img.shape[1]}x{img.shape[0]}")
        result = ocr.read(img)
        if result:
            normalized = normalize_indian_plate(result.text)
            print(f"  Raw OCR:    '{result.text}'")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Normalized: {normalized or '(failed normalization)'}")
        else:
            print("  No text recognized.")
        return

    if args.synthetic:
        test_plates = ["TS09EA1234", "AP31AB9999", "MH02CD5678", "KA05MJ1234", "DL3CAB1234"]
        passed = 0
        for plate in test_plates:
            img = make_synthetic_plate(plate)
            if args.save:
                cv2.imwrite(args.save, img)
            result = ocr.read(img)
            if result:
                normalized = normalize_indian_plate(result.text)
                ok = normalized == plate
                marker = "[OK]" if ok else "[FAIL]"
                print(f"  {marker} expected={plate}  raw='{result.text}'  norm='{normalized}'  conf={result.confidence:.2f}")
                if ok:
                    passed += 1
            else:
                print(f"  [FAIL] expected={plate}  (no text recognized)")
        print(f"\n{passed}/{len(test_plates)} plates recognized correctly.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
