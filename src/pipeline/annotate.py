"""Render an annotated frame for evidence — boxes, labels, stop line."""
from __future__ import annotations

import cv2
import numpy as np

from src.detection.classes import ObjectClass
from src.detection.detector import DetectionBundle
from src.tracking.tracker import TrackedDetection

CLASS_COLOR = {
    ObjectClass.PERSON: (0, 255, 255),
    ObjectClass.TWO_WHEELER: (0, 200, 255),
    ObjectClass.CAR: (0, 255, 0),
    ObjectClass.BUS: (0, 255, 0),
    ObjectClass.TRUCK: (0, 255, 0),
    ObjectClass.LICENSE_PLATE: (255, 200, 0),
    ObjectClass.HELMET: (0, 255, 100),
    ObjectClass.NO_HELMET: (0, 0, 255),
    ObjectClass.TRAFFIC_LIGHT: (200, 200, 200),
}


def annotate(
    frame: np.ndarray,
    bundle: DetectionBundle,
    tracked: list[TrackedDetection],
    stop_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    signal_state: str | None = None,
) -> np.ndarray:
    out = frame.copy()
    for d in bundle.detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        color = CLASS_COLOR.get(d.cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d.cls.value} {d.conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for td in tracked:
        x1, y1, x2, y2 = map(int, td.detection.xyxy)
        cv2.putText(out, f"#{td.track_id}", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if stop_line is not None:
        cv2.line(out, stop_line[0], stop_line[1], (0, 0, 255), 2)

    if signal_state:
        cv2.putText(out, f"signal: {signal_state}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out
