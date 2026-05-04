"""Seed the database with synthetic demo violations.

Generates sample frames + plate crops + violations so you can see the
dashboard with realistic data. Use this to verify the UI before plugging
in a real CCTV feed.

Usage:
    python scripts/seed_demo.py
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.db import engine, session_scope
from src.evidence.models import Base, CameraHealth
from src.evidence.store import EvidenceStore
from src.rules.types import ViolationCode, ViolationEvent


SAMPLE_VIOLATIONS = [
    # (code, camera, plate, rule_conf, ocr_conf)
    (ViolationCode.NO_HELMET, "CAM01", "TS09EA1234", 0.92, 0.95),
    (ViolationCode.NO_HELMET, "CAM01", "AP31AB9999", 0.88, 0.91),
    (ViolationCode.RED_LIGHT_JUMP, "CAM01", "MH02CD5678", 0.95, 0.89),
    (ViolationCode.NO_HELMET, "CAM02", "KA05MJ1234", 0.87, 0.93),
    (ViolationCode.PLATE_UNREADABLE, "CAM02", None, 0.45, 0.20),
    (ViolationCode.RED_LIGHT_JUMP, "CAM01", "DL3CAB5678", 0.91, 0.85),
    (ViolationCode.NO_HELMET, "CAM03", "GJ01AA1234", 0.83, 0.88),
    (ViolationCode.NO_HELMET, "CAM01", "TN09BB2222", 0.89, 0.92),
]


def make_synthetic_frame(plate_text: str | None, code: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (full_frame, annotated_frame, plate_crop)."""
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (0, 500), (1280, 720), (60, 60, 60), -1)
    cv2.line(frame, (100, 600), (1180, 600), (255, 255, 255), 3)

    veh_x, veh_y = 500, 380
    cv2.rectangle(frame, (veh_x, veh_y), (veh_x + 280, veh_y + 200), (180, 100, 50), -1)
    cv2.rectangle(frame, (veh_x + 30, veh_y + 30), (veh_x + 250, veh_y + 80), (40, 40, 40), -1)
    cv2.putText(frame, "MH 14", (veh_x + 80, veh_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    plate_w, plate_h = 200, 50
    px = veh_x + 40
    py = veh_y + 230
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (255, 255, 255), -1)
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (0, 0, 0), 2)
    if plate_text:
        cv2.putText(frame, plate_text, (px + 10, py + 35), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)

    annotated = frame.copy()
    cv2.rectangle(annotated, (veh_x, veh_y), (veh_x + 280, veh_y + 200), (0, 0, 255), 3)
    cv2.putText(annotated, code.upper().replace("_", " "),
                (veh_x, veh_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.rectangle(annotated, (px - 2, py - 2), (px + plate_w + 2, py + plate_h + 2), (0, 255, 255), 2)

    plate_crop = frame[py:py + plate_h, px:px + plate_w].copy()
    return frame, annotated, plate_crop


def main():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine())

    print("Seeding camera health...")
    with session_scope() as s:
        for cam in ["CAM01", "CAM02", "CAM03"]:
            existing = s.get(CameraHealth, cam)
            if existing is None:
                s.add(CameraHealth(camera_id=cam, last_frame_at=datetime.utcnow(),
                                   fps_observed=14.5, is_up=True))

    store = EvidenceStore()
    print(f"Inserting {len(SAMPLE_VIOLATIONS)} sample violations...\n")

    base_time = time.time() - 3600
    for i, (code, cam, plate, rule_conf, ocr_conf) in enumerate(SAMPLE_VIOLATIONS):
        frame, annotated, crop = make_synthetic_frame(plate, code.value)
        event = ViolationEvent(
            code=code,
            camera_id=cam,
            track_id=100 + i,
            timestamp=base_time + i * 300,
            frame_idx=1000 + i,
            confidence=rule_conf,
            plate_text=plate,
            plate_ocr_confidence=ocr_conf,
            bbox=(500, 380, 780, 580),
        )
        vid = store.save(event, frame=frame, annotated=annotated, plate_crop=crop)
        print(f"  [{i+1}/{len(SAMPLE_VIOLATIONS)}] id={vid}  cam={cam}  code={code.value}  plate={plate or '-'}")

    print("\nDone. Start the API server:")
    print("    python scripts/run_api.py")
    print("Then open http://localhost:8000/")


if __name__ == "__main__":
    main()
