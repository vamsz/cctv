"""Extract plate crops from sample videos for hand-labelling.

Workflow:
  1. Run this script — it scans data/samples/*.mp4, detects plates
     using the fine-tuned plate_ft.pt + fast-alpr, and saves each
     crop to data/plate_ocr_ft/raw/ named <auto_ocr_guess>.jpg.
  2. Open data/plate_ocr_ft/raw/ in any file browser.
  3. For each image, look at the actual plate and RENAME the file to
     the correct text. e.g. "AP02NN9091.jpg" → if it's actually
     "AP02MN9091" rename to "AP02MN9091.jpg".
  4. Move/copy the renamed files to data/plate_ocr_ft/labelled/.
  5. Run scripts/train_plate_recognizer.py to fine-tune.

Why hand-labelling: even a few hundred labelled Indian plate crops
beat any synthetic dataset for our specific cameras / fonts.

Tip: skip files that are too blurry or partially occluded — they
add noise to training. Aim for 300-1000 clean labels.

Usage:
  python scripts/extract_plate_crops_for_labelling.py
  python scripts/extract_plate_crops_for_labelling.py --source data/samples/test2.mp4
  python scripts/extract_plate_crops_for_labelling.py --max-per-video 100
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Suppress noisy boot messages
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_plates")

OUT_DIR = ROOT / "data" / "plate_ocr_ft" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(text: str) -> str:
    """Make a filesystem-safe filename from an OCR string."""
    s = "".join(c for c in text.upper() if c.isalnum())
    return s if s else "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "samples",
                    help="single video file OR directory of videos")
    ap.add_argument("--max-per-video", type=int, default=200,
                    help="cap the number of crops saved per video (default 200)")
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="process every Nth frame (default 5)")
    ap.add_argument("--min-side", type=int, default=40,
                    help="skip crops with either side smaller than this")
    args = ap.parse_args()

    if args.source.is_file():
        videos = [args.source]
    else:
        videos = sorted(args.source.glob("*.mp4"))
    if not videos:
        log.error("no .mp4 files found at %s", args.source)
        sys.exit(1)

    log.info("loading plate detector + OCR ...")
    from src.detection.detector import Detector
    from src.ocr.plate_ocr import PlateOCR
    from config.settings import settings

    det = Detector(
        general_weights=settings.detector_weights,
        helmet_weights=settings.helmet_weights,
        plate_weights=settings.plate_weights,
        device=settings.device,
        conf=0.30,                     # lower than runtime — we want recall
        use_fast_alpr=True,
    )
    ocr = PlateOCR(use_gpu=False)
    log.info("ready. writing crops to %s", OUT_DIR)

    total_saved = 0
    for video in videos:
        log.info("scanning %s", video.name)
        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            log.warning("  bad video, skipping")
            cap.release()
            continue

        per_video_saved = 0
        frame_idx = 0
        seen_texts: set[str] = set()
        t0 = time.time()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.frame_stride != 0:
                continue
            if per_video_saved >= args.max_per_video:
                break

            bundle = det(frame, frame_idx, time.time())
            from src.detection.classes import ObjectClass
            for plate in bundle.of(ObjectClass.LICENSE_PLATE):
                x1, y1, x2, y2 = plate.xyxy
                w, h = x2 - x1, y2 - y1
                if w < args.min_side or h < 0.6 * args.min_side:
                    continue
                # 25% pad for context
                from src.pipeline.runner import _crop_padded
                crop = _crop_padded(frame, plate.xyxy, pad=0.25)
                if crop is None or crop.size == 0:
                    continue

                # Get the OCR's best guess for the filename
                read = ocr.read(crop)
                guess = read.text if read else "UNREAD"
                fname_base = safe_filename(guess)
                # Dedup by guess so a slow-moving car doesn't fill the folder
                # with 60 copies of the same plate.
                if fname_base in seen_texts:
                    continue
                seen_texts.add(fname_base)

                # Save with unique suffix in case multiple plates share
                # the same OCR guess across different videos
                out_name = f"{video.stem}_{frame_idx:06d}_{fname_base}.jpg"
                cv2.imwrite(str(OUT_DIR / out_name), crop)
                per_video_saved += 1
                total_saved += 1
                if per_video_saved <= 5 or per_video_saved % 25 == 0:
                    log.info("  saved %d crops (latest=%s shape=%dx%d)",
                             per_video_saved, fname_base, w, h)

        cap.release()
        log.info("  %s done: %d crops in %.1fs", video.name, per_video_saved, time.time() - t0)

    log.info("=" * 70)
    log.info("Extracted %d total plate crops to %s", total_saved, OUT_DIR)
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Open %s in your file browser", OUT_DIR)
    log.info("  2. For each image, look at the plate and rename if OCR was wrong:")
    log.info("     bad_guess.jpg  →  CORRECT_PLATE.jpg")
    log.info("  3. Skip / delete blurry / partial plates")
    log.info("  4. Move correctly-named files to data/plate_ocr_ft/labelled/")
    log.info("  5. Run: python scripts/train_plate_recognizer.py")


if __name__ == "__main__":
    main()
