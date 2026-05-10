"""Render annotated evidence frames — boxes, labels, stop line, violation banners."""
from __future__ import annotations

import cv2
import numpy as np

from src.detection.classes import ObjectClass
from src.detection.detector import DetectionBundle
from src.tracking.tracker import TrackedDetection

CLASS_COLOR = {
    ObjectClass.PERSON: (0, 220, 220),
    ObjectClass.TWO_WHEELER: (0, 190, 255),
    ObjectClass.CAR: (80, 200, 80),
    ObjectClass.BUS: (80, 200, 80),
    ObjectClass.TRUCK: (80, 200, 80),
    ObjectClass.LICENSE_PLATE: (200, 160, 0),
    ObjectClass.HELMET: (0, 220, 80),
    ObjectClass.NO_HELMET: (0, 50, 220),
    ObjectClass.TRAFFIC_LIGHT: (180, 180, 180),
}

VIOLATION_COLOR = {
    "no_helmet": (0, 50, 220),
    "red_light_jump": (0, 30, 200),
    "wrong_way": (0, 0, 255),
    "triple_riding": (20, 100, 200),
    "overspeed": (50, 120, 220),
    "plate_unreadable": (140, 90, 0),
    "watchlist_hit": (160, 0, 180),
}


def annotate(
    frame: np.ndarray,
    bundle: DetectionBundle,
    tracked: list[TrackedDetection],
    stop_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    signal_state: str | None = None,
    violations: list | None = None,
) -> np.ndarray:
    out = frame.copy()

    for d in bundle.detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        color = CLASS_COLOR.get(d.cls, (200, 200, 200))

        # Plates get extra-prominent treatment: thicker box + OCR text
        if d.cls == ObjectClass.LICENSE_PLATE:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if d.text:
                # Draw a filled background strip ABOVE the plate with the OCR text
                label = d.text.strip()
                conf = d.text_conf if d.text_conf is not None else d.conf
                strip = f"{label}  {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(strip, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
                bx1, by1 = x1, max(0, y1 - th - 8)
                bx2, by2 = min(out.shape[1], x1 + tw + 12), y1
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 165, 255), -1)
                cv2.putText(out, strip, (bx1 + 6, by2 - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA)
            else:
                cv2.putText(out, f"plate {d.conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, f"{d.cls.value} {d.conf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    for td in tracked:
        x1, y1, x2, y2 = map(int, td.detection.xyxy)
        cv2.putText(out, f"#{td.track_id}", (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if stop_line is not None:
        cv2.line(out, stop_line[0], stop_line[1], (0, 0, 220), 2)

    if signal_state:
        sig_color = {
            "RED": (50, 50, 220), "GREEN": (50, 200, 50), "YELLOW": (50, 180, 200)
        }.get(signal_state.upper(), (200, 200, 200))
        cv2.rectangle(out, (8, 8), (180, 34), (0, 0, 0), -1)
        cv2.putText(out, f"SIGNAL: {signal_state.upper()}", (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sig_color, 1, cv2.LINE_AA)

    # Draw violation banners for each active event
    if violations:
        for i, ev in enumerate(violations):
            vcolor = VIOLATION_COLOR.get(ev.code.value if hasattr(ev.code, 'value') else ev.code, (200, 50, 50))
            if ev.bbox:
                x1, y1, x2, y2 = map(int, ev.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), vcolor, 3)
                label = str(ev.code.value if hasattr(ev.code, 'value') else ev.code).upper().replace("_", " ")
                cv2.rectangle(out, (x1, y1 - 22), (x2, y1), vcolor, -1)
                cv2.putText(out, label, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out
