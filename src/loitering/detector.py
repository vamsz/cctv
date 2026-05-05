"""Loitering and abandoned-object detection.

LoiteringDetector
  Tracks how many consecutive frames each person track spends inside
  each configured zone (foot-point containment via pointPolygonTest).
  Fires a LoiterEvent when dwell exceeds the zone threshold, then re-fires
  every `refine_interval_s` seconds so the caller can update the DB row.

AbandonedObjectDetector
  MOG2 background subtraction → foreground blobs → static blobs that
  have no person overlap for min_frames → AbandonedObject.
  Runs on a 2× downsampled frame for speed (~3ms at 720p).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class LoiterEvent:
    camera_id: str
    track_id: int
    zone_id: str
    zone_label: str
    dwell_frames: int
    dwell_seconds: float
    bbox: Optional[tuple]   # (x1,y1,x2,y2) current position


@dataclass
class AbandonedObject:
    bbox: tuple             # (x1,y1,x2,y2) native resolution
    age_frames: int
    centroid: tuple         # (cx, cy)


class LoiteringDetector:
    def __init__(
        self,
        camera_id: str,
        zones: list[dict],
        fps: float = 15.0,
        refine_interval_s: float = 5.0,
    ):
        """
        zones: list of dicts with keys:
          id, label, polygon [[x,y],...], threshold_seconds
        """
        self.camera_id = camera_id
        self.fps = fps
        self._refine_frames = max(1, int(refine_interval_s * fps))

        self._zones: list[dict] = []
        for z in zones:
            self._zones.append({
                "id": z["id"],
                "label": z.get("label", z["id"]),
                "polygon": np.array(z["polygon"], dtype=np.int32),
                "threshold_frames": max(1, int(z.get("threshold_seconds", 30) * fps)),
            })

        # track_id → {zone_id: frames_in_zone}
        self._dwell: dict[int, dict[str, int]] = {}
        self._bboxes: dict[int, tuple] = {}

    # ------------------------------------------------------------------

    def update(self, tracked_persons: list[tuple]) -> list[LoiterEvent]:
        """
        tracked_persons: list of (track_id, x1, y1, x2, y2)
        """
        active_ids: set[int] = set()
        events: list[LoiterEvent] = []

        for tid, x1, y1, x2, y2 in tracked_persons:
            active_ids.add(tid)
            foot = (float((x1 + x2) / 2), float(y2))
            self._bboxes[tid] = (x1, y1, x2, y2)

            if tid not in self._dwell:
                self._dwell[tid] = {}

            for z in self._zones:
                in_zone = cv2.pointPolygonTest(
                    z["polygon"].reshape(-1, 1, 2), foot, False
                ) >= 0

                if in_zone:
                    self._dwell[tid][z["id"]] = self._dwell[tid].get(z["id"], 0) + 1
                    frames = self._dwell[tid][z["id"]]
                    # Fire once threshold crossed, then every refine_frames
                    if frames >= z["threshold_frames"] and (
                        frames == z["threshold_frames"]
                        or frames % self._refine_frames == 0
                    ):
                        events.append(LoiterEvent(
                            camera_id=self.camera_id,
                            track_id=tid,
                            zone_id=z["id"],
                            zone_label=z["label"],
                            dwell_frames=frames,
                            dwell_seconds=round(frames / self.fps, 1),
                            bbox=self._bboxes.get(tid),
                        ))
                else:
                    self._dwell[tid].pop(z["id"], None)

        # clean up disappeared tracks
        for tid in [t for t in self._dwell if t not in active_ids]:
            self._dwell.pop(tid, None)
            self._bboxes.pop(tid, None)

        return events


class AbandonedObjectDetector:
    """MOG2-based abandoned-object detector."""

    def __init__(
        self,
        min_frames: int = 150,
        min_area_px2: int = 2500,
        person_iou_thresh: float = 0.15,
    ):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=False
        )
        self.min_frames = min_frames
        self.min_area = min_area_px2
        self.person_iou_thresh = person_iou_thresh
        # quantised centroid key → age (frames)
        self._blobs: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        person_bboxes: list[tuple],
    ) -> list[AbandonedObject]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        mask = self._mog2.apply(small)

        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, ker)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        live_keys: set[tuple[int, int]] = set()
        out: list[AbandonedObject] = []

        for cnt in contours:
            area = cv2.contourArea(cnt) * 4   # scale back to native px²
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x, y, bw, bh = x * 2, y * 2, bw * 2, bh * 2
            bbox = (x, y, x + bw, y + bh)
            cx, cy = x + bw // 2, y + bh // 2
            key = (cx // 20 * 20, cy // 20 * 20)
            live_keys.add(key)

            self._blobs[key] = self._blobs.get(key, 0) + 1
            age = self._blobs[key]

            if age < self.min_frames:
                continue

            near_person = any(_iou(bbox, pb) > self.person_iou_thresh for pb in person_bboxes)
            if not near_person:
                out.append(AbandonedObject(bbox=bbox, age_frames=age, centroid=(cx, cy)))

        # expire stale blobs
        for k in [k for k in self._blobs if k not in live_keys]:
            self._blobs.pop(k, None)

        return out


def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)
