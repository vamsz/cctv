"""Quick test: run detection + tracking + OCR + violence detection on a video file.

Usage:
    python scripts/test_video.py --video path/to/your_video.mp4
    python scripts/test_video.py --video path/to/your_video.mp4 --save output.mp4
    python scripts/test_video.py --video 0   (for webcam)

Controls:
    q     quit
    space pause/resume
    s     save current frame as screenshot

Supports any plate format: Indian, UK, EU, US, etc.
"""
import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.detection.detector import Detector
from src.ocr.plate_normalize import normalize_plate
from src.ocr.plate_ocr import PlateOCR
from src.tracking.tracker import Tracker
from src.rules.geometry import contains
from src.violence.clip_classifier import ViolenceClipClassifier


CLASS_COLORS = {
    ObjectClass.CAR: (255, 180, 50),
    ObjectClass.TRUCK: (255, 100, 50),
    ObjectClass.BUS: (255, 50, 100),
    ObjectClass.TWO_WHEELER: (50, 255, 180),
    ObjectClass.PERSON: (200, 200, 200),
    ObjectClass.TRAFFIC_LIGHT: (0, 200, 255),
    ObjectClass.LICENSE_PLATE: (0, 255, 255),
    ObjectClass.HELMET: (0, 255, 0),
    ObjectClass.NO_HELMET: (0, 0, 255),
}

VIOLENCE_COLOR_OK   = (0, 220, 0)    # green — low score
VIOLENCE_COLOR_MED  = (0, 165, 255)  # orange — medium
VIOLENCE_COLOR_HIGH = (0, 0, 255)    # red — confirmed fight


def draw(frame, tracked, plate_reads, bundle, violence_score: float | None):
    for td in tracked:
        x1, y1, x2, y2 = (int(v) for v in td.detection.xyxy)
        cls = td.detection.cls
        color = CLASS_COLORS.get(cls, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{td.track_id} {cls.name} {td.detection.conf:.2f}"
        if td.track_id in plate_reads:
            text, conf = plate_reads[td.track_id]
            label += f" | {text} ({conf:.2f})"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for det in bundle.of(ObjectClass.LICENSE_PLATE):
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Violence score bar (top-right corner)
    if violence_score is not None:
        H, W = frame.shape[:2]
        bar_w, bar_h = 220, 28
        bx = W - bar_w - 10
        by = 10
        pct = int(bar_w * min(violence_score, 1.0))
        if violence_score >= 0.55:
            col = VIOLENCE_COLOR_HIGH
        elif violence_score >= 0.35:
            col = VIOLENCE_COLOR_MED
        else:
            col = VIOLENCE_COLOR_OK
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, by), (bx + pct, by + bar_h), col, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 1)
        tag = f"FIGHT {violence_score:.2f}" if violence_score >= 0.55 else f"Violence {violence_score:.2f}"
        cv2.putText(frame, tag, (bx + 6, by + bar_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    ap = argparse.ArgumentParser(description="Test CCTV detection on a video file")
    ap.add_argument("--video", required=True, help="Path to video file, RTSP URL, or webcam index (0)")
    ap.add_argument("--save", default=None, help="Save annotated output (e.g. output.mp4). Must differ from --video.")
    ap.add_argument("--device", default=None, help="Override device (cpu, cuda:0)")
    ap.add_argument("--conf", type=float, default=None, help="Override detection confidence threshold")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR (faster)")
    ap.add_argument("--no-violence", action="store_true", help="Skip violence detection")
    ap.add_argument("--no-display", action="store_true", help="Run headless (useful with --save)")
    ap.add_argument("--fast-ocr", action="store_true", help="Use 2 OCR passes instead of 6 (for CPU)")
    ap.add_argument("--every-n", type=int, default=1, help="Process only every N-th frame")
    ap.add_argument("--debug-crops", action="store_true", help="Save plate crops to data/debug_crops/")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 = no limit)")
    args = ap.parse_args()

    # Guard: don't overwrite the source file
    if args.save and not args.video.isdigit():
        src_path = Path(args.video).resolve()
        dst_path = Path(args.save).resolve()
        if src_path == dst_path:
            print(f"ERROR: --save cannot be the same file as --video.\n"
                  f"  Use a different output path, e.g.:\n"
                  f"  --save data/samples/test_fight_out.mp4")
            sys.exit(1)

    debug_dir = Path("data/debug_crops")
    if args.debug_crops:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for f in debug_dir.glob("*.jpg"):
            f.unlink()

    device = args.device or settings.device
    conf = args.conf or settings.det_conf

    print(f"Loading models on {device} (conf={conf})...")
    detector = Detector(
        general_weights=settings.detector_weights,
        helmet_weights=settings.helmet_weights,
        plate_weights=settings.plate_weights,
        device=device,
        conf=conf,
    )

    ocr = None
    if not args.no_ocr:
        print("Loading OCR...")
        fast = args.fast_ocr or device == "cpu"
        ocr = PlateOCR(lang=settings.ocr_lang, use_gpu=device.startswith("cuda"), fast_mode=fast)
        print(f"  OCR mode: {'fast (2 passes)' if fast else 'accurate (6 passes)'}")

    violence_clf = None
    latest_violence: dict = {"score": None, "label": ""}
    v_lock = threading.Lock()

    if not args.no_violence:
        print("Loading violence classifier...")
        violence_clf = ViolenceClipClassifier(device=device, threshold=0.55)
        if violence_clf.available:
            print("  Violence classifier ready")
        else:
            print("  Violence classifier unavailable (torchvision not installed)")
            violence_clf = None

    def _on_violence_result(result):
        with v_lock:
            latest_violence["score"] = result.score
            latest_violence["label"] = result.label
        tag = "FIGHT CONFIRMED" if result.confirmed else "non-fight"
        print(f"  [Violence] score={result.score:.3f} ({tag}) label={result.label}")

    tracker = Tracker(frame_rate=15)
    plate_reads: dict[int, tuple[str, float]] = {}

    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"Video: {W}x{H}, {total_frames} frames, {fps:.1f} fps")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))
        print(f"Saving annotated output to {args.save}")

    frame_idx = 0
    paused = False
    t_start = time.time()
    det_count = 0
    # Request violence inference every N frames once buffer has enough data
    _VIOLENCE_INTERVAL = 16

    print("\nRunning... (press 'q' to quit, 'space' to pause)\n")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print(f"\nDone -- processed {frame_idx} frames")
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                print(f"\nReached --max-frames {args.max_frames}")
                break

            if args.every_n > 1 and frame_idx % args.every_n != 0:
                if writer:
                    writer.write(frame)
                continue

            ts = time.time()
            bundle = detector(frame, frame_idx, ts)
            tracked = tracker.update(bundle)
            n_plates = len(bundle.of(ObjectClass.LICENSE_PLATE))

            # Violence: feed every frame; request inference periodically
            if violence_clf is not None:
                violence_clf.submit_frame("test", frame)
                if frame_idx % _VIOLENCE_INTERVAL == 0:
                    violence_clf.request_inference("test", _on_violence_result)

            elapsed = time.time() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)

            with v_lock:
                vscore = latest_violence["score"]

            print(f"[{progress}] {proc_fps:.1f} fps | {len(tracked)} tracked | "
                  f"{n_plates} plates detected | {det_count} read"
                  + (f" | violence={vscore:.2f}" if vscore is not None else ""), flush=True)

            plate_dets = bundle.of(ObjectClass.LICENSE_PLATE)
            for td in tracked:
                if td.detection.cls not in VEHICLE_CLASSES:
                    continue
                if td.track_id in plate_reads:
                    continue
                for p in plate_dets:
                    if contains(td.detection.xyxy, p.xyxy) >= 0.5:
                        if ocr:
                            x1, y1, x2, y2 = (max(0, int(v)) for v in p.xyxy)
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size == 0:
                                break
                            ch, cw = crop.shape[:2]
                            # Skip crops that don't have plate-like aspect ratio
                            if ch > 0 and cw / ch < 1.4:
                                print(f"  PLATE Track#{td.track_id} crop={cw}x{ch} "
                                      f"det_conf={p.conf:.2f} -> skipped (not plate shape)", flush=True)
                                break
                            read = ocr.read(crop)
                            raw = read.text if read else None
                            conf_val = read.confidence if read else 0.0
                            normalized = normalize_plate(raw) if raw else None
                            display = normalized or raw or "(no text)"
                            if read and read.text:
                                plate_reads[td.track_id] = (display, conf_val)
                                det_count += 1
                            print(f"  PLATE Track#{td.track_id} crop={cw}x{ch} "
                                  f"det_conf={p.conf:.2f} -> raw='{raw or '(empty)'}' "
                                  f"normalized='{normalized or '-'}' ocr_conf={conf_val:.2f}", flush=True)
                            if args.debug_crops:
                                label = (raw or "empty").replace("/", "_").replace(":", "_")[:20]
                                cv2.imwrite(str(debug_dir / f"f{frame_idx:04d}_t{td.track_id}_{label}.jpg"), crop)
                        break

            with v_lock:
                vscore = latest_violence["score"]

            annotated = draw(frame.copy(), tracked, plate_reads, bundle, vscore)

            elapsed = time.time() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            info = (f"Frame {frame_idx}/{total_frames} | {proc_fps:.1f} fps | "
                    f"{len(tracked)} tracked | {det_count} plates read")
            cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            if writer:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("CCTV Test", annotated)

        if not args.no_display:
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord("s"):
                out_path = f"data/screenshot_{frame_idx}.jpg"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(out_path, annotated)
                print(f"Saved screenshot: {out_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nSummary:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Time: {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
    print(f"  Plates read: {det_count}")
    if plate_reads:
        print(f"  Plate texts:")
        for tid, (text, conf_v) in sorted(plate_reads.items()):
            print(f"    Track #{tid}: {text} (confidence: {conf_v:.2f})")
    with v_lock:
        vscore = latest_violence["score"]
        vlabel = latest_violence["label"]
    if vscore is not None:
        print(f"  Final violence score: {vscore:.3f} (last label: {vlabel})")


if __name__ == "__main__":
    main()
