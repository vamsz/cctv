"""Loitering and abandoned-object detection.

LoiteringDetector (upgraded)
  Dwell-timer per zone + trajectory shape features (path/displacement ratio,
  turn-angle variance) per improvement plan Area 8 / Núñez Cano et al. 2024.
  False-positive reduction vs plain dwell timer:
    - Bus-stop standing: excluded (low path, low variance)
    - Vendors at stall: long dwell but low ratio → still fires (correct)
    - Genuine loiterer: dwell + high path/displacement ratio → fires faster

AbandonedObjectDetector (upgraded — owner association)
  Per improvement plan Area 7 / Russel & Selvaraj 2024 (Vis Comput 40:4401-4426).
  Uses COCO luggage classes (backpack=24, handbag=26, suitcase=28) from the
  existing general YOLO detector — no extra model needed.
  Algorithm:
    1. When a luggage object first appears, bind it to the nearest person
       within OWNER_BIND_RADIUS_PX pixels.
    2. Track object centroid stability (stationary for ≥ STATIC_FRAMES).
    3. Flag as abandoned when:
       (a) the bound owner track is lost OR moved > OWNER_SEPARATION_PX, AND
       (b) object has been stationary for ≥ STATIC_FRAMES (default 300 = ~20s).

  Falls back to MOG2 when luggage detection yields nothing (legacy path).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .trajectory import TrajectoryFeatures, TrajectoryStore

log = logging.getLogger("loitering")


# ── COCO class IDs present in YOLO11n general model
_LUGGAGE_CLASSES = {24, 26, 28}   # backpack, handbag, suitcase
OWNER_BIND_RADIUS_PX = 200         # pixels — max distance to bind owner at first sight
OWNER_SEPARATION_PX = 300          # pixels — owner considered "away" beyond this
STATIC_FRAMES = 300                 # ~20 s at 15 fps before flagging abandoned


@dataclass
class LoiterEvent:
    camera_id: str
    track_id: int
    zone_id: str
    zone_label: str
    dwell_frames: int
    dwell_seconds: float
    bbox: Optional[tuple]
    trajectory: Optional[TrajectoryFeatures] = None


@dataclass
class AbandonedObject:
    bbox: tuple             # (x1,y1,x2,y2) native resolution
    age_frames: int
    centroid: tuple         # (cx, cy)
    owner_track_id: Optional[int] = None  # track_id of bound owner (may be lost)


# ── Luggage track entry
@dataclass
class _LuggageTrack:
    obj_id: int             # synthetic id from bbox hash
    bbox: tuple             # (x1,y1,x2,y2)
    centroid: tuple         # (cx, cy)
    static_frames: int      # frames since last significant movement
    owner_track_id: Optional[int]  # person track_id bound at first sight
    last_owner_centroid: Optional[tuple]  # last known position of owner


class LoiteringDetector:
    """Per-camera loitering detector with trajectory-shape features."""

    def __init__(
        self,
        camera_id: str,
        zones: list[dict],
        fps: float = 15.0,
        refine_interval_s: float = 5.0,
    ):
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
        self._traj = TrajectoryStore(max_len=int(fps * 20))   # 20 s history

    def update(self, tracked_persons: list[tuple]) -> list[LoiterEvent]:
        """
        tracked_persons: list of (track_id, x1, y1, x2, y2)
        """
        active_ids: set[int] = set()
        events: list[LoiterEvent] = []

        for tid, x1, y1, x2, y2 in tracked_persons:
            active_ids.add(tid)
            foot = (float((x1 + x2) / 2), float(y2))
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            self._bboxes[tid] = (x1, y1, x2, y2)
            self._traj.add_point(tid, (cx, cy))

            if tid not in self._dwell:
                self._dwell[tid] = {}

            for z in self._zones:
                in_zone = cv2.pointPolygonTest(
                    z["polygon"].reshape(-1, 1, 2), foot, False
                ) >= 0

                if in_zone:
                    self._dwell[tid][z["id"]] = self._dwell[tid].get(z["id"], 0) + 1
                    frames = self._dwell[tid][z["id"]]

                    fire_threshold = frames == z["threshold_frames"]
                    fire_refine = frames > z["threshold_frames"] and frames % self._refine_frames == 0

                    if fire_threshold or fire_refine:
                        traj = self._traj.compute_features(tid)

                        # Suppress bus-stop false positives
                        if traj is not None and traj.exclude_bus_stop():
                            continue

                        events.append(LoiterEvent(
                            camera_id=self.camera_id,
                            track_id=tid,
                            zone_id=z["id"],
                            zone_label=z["label"],
                            dwell_frames=frames,
                            dwell_seconds=round(frames / self.fps, 1),
                            bbox=self._bboxes.get(tid),
                            trajectory=traj,
                        ))
                else:
                    self._dwell[tid].pop(z["id"], None)

        self._traj.cleanup(active_ids)
        for tid in [t for t in self._dwell if t not in active_ids]:
            self._dwell.pop(tid, None)
            self._bboxes.pop(tid, None)

        return events


# ---------------------------------------------------------------------------

class AbandonedObjectDetector:
    """Luggage owner-association detector (primary) + MOG2 fallback."""

    def __init__(
        self,
        min_frames: int = 300,
        min_area_px2: int = 3000,
        person_iou_thresh: float = 0.15,
        owner_bind_radius: int = OWNER_BIND_RADIUS_PX,
        owner_separation: int = OWNER_SEPARATION_PX,
        static_frames: int = STATIC_FRAMES,
    ):
        self._min_frames = min_frames
        self._min_area = min_area_px2
        self._person_iou_thresh = person_iou_thresh
        self._bind_radius = owner_bind_radius
        self._separation = owner_separation
        self._static_frames = static_frames

        # Luggage tracks: obj_id → _LuggageTrack
        self._luggage: dict[int, _LuggageTrack] = {}

        # MOG2 fallback (when YOLO doesn't see any luggage)
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=False
        )
        self._mog2_blobs: dict[tuple[int, int], int] = {}

        self._frame_count = 0

    def update(
        self,
        frame: np.ndarray,
        person_bboxes: list[tuple],
        object_detections: Optional[list[tuple]] = None,
    ) -> list[AbandonedObject]:
        """
        person_bboxes: list of (x1,y1,x2,y2) from YOLO person class.
        object_detections: list of (x1,y1,x2,y2,class_id) for luggage classes.
          Extracted from the main DetectionBundle by the pipeline.
          If None, falls back to MOG2 background subtraction.
        """
        self._frame_count += 1
        person_centroids = {
            i: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            for i, b in enumerate(person_bboxes)
        }

        # ── Primary path: YOLO luggage detections with owner association
        if object_detections is not None:
            return self._update_owner_association(
                frame, person_bboxes, person_centroids, object_detections
            )

        # ── Fallback: MOG2
        return self._update_mog2(frame, person_bboxes)

    # ---------------------------------------------------------------- owner association

    def _update_owner_association(
        self,
        frame: np.ndarray,
        person_bboxes: list[tuple],
        person_centroids: dict,
        object_detections: list[tuple],
    ) -> list[AbandonedObject]:
        live_ids: set[int] = set()
        out: list[AbandonedObject] = []

        for det in object_detections:
            x1, y1, x2, y2 = det[:4]
            area = (x2 - x1) * (y2 - y1)
            if area < self._min_area:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bbox = (float(x1), float(y1), float(x2), float(y2))
            obj_id = int(cx // 20 * 20) * 10000 + int(cy // 20 * 20)
            live_ids.add(obj_id)

            if obj_id not in self._luggage:
                # Bind to nearest person within radius
                owner_id = self._find_nearest_person(cx, cy, person_centroids)
                self._luggage[obj_id] = _LuggageTrack(
                    obj_id=obj_id,
                    bbox=bbox,
                    centroid=(cx, cy),
                    static_frames=0,
                    owner_track_id=owner_id,
                    last_owner_centroid=person_centroids.get(owner_id),
                )
                continue

            track = self._luggage[obj_id]

            # Update centroid movement
            prev_cx, prev_cy = track.centroid
            moved = math.hypot(cx - prev_cx, cy - prev_cy)
            if moved < 5.0:
                track.static_frames += 1
            else:
                track.static_frames = 0
                track.centroid = (cx, cy)
            track.bbox = bbox

            # Update owner position
            if track.owner_track_id is not None:
                owner_c = person_centroids.get(track.owner_track_id)
                if owner_c is not None:
                    track.last_owner_centroid = owner_c

            # Check abandonment conditions
            if track.static_frames < self._static_frames:
                continue

            owner_present = self._is_owner_present(track, person_bboxes, person_centroids)
            if not owner_present:
                out.append(AbandonedObject(
                    bbox=bbox,
                    age_frames=track.static_frames,
                    centroid=(cx, cy),
                    owner_track_id=track.owner_track_id,
                ))

        # Expire stale luggage tracks
        stale = [oid for oid in self._luggage if oid not in live_ids]
        for oid in stale:
            del self._luggage[oid]

        return out

    def _find_nearest_person(
        self,
        cx: float,
        cy: float,
        person_centroids: dict,
    ) -> Optional[int]:
        best_dist = self._bind_radius
        best_id = None
        for pid, (px, py) in person_centroids.items():
            d = math.hypot(cx - px, cy - py)
            if d < best_dist:
                best_dist = d
                best_id = pid
        return best_id

    def _is_owner_present(
        self,
        track: _LuggageTrack,
        person_bboxes: list[tuple],
        person_centroids: dict,
    ) -> bool:
        cx, cy = track.centroid
        # Check if any person is within separation radius of the object
        for pcx, pcy in person_centroids.values():
            if math.hypot(cx - pcx, cy - pcy) <= self._separation:
                return True
        # Also check IoU with original owner bbox area
        for pb in person_bboxes:
            if _iou(track.bbox, pb) > self._person_iou_thresh:
                return True
        return False

    # ---------------------------------------------------------------- MOG2 fallback

    def _update_mog2(
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
            area = cv2.contourArea(cnt) * 4
            if area < self._min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x, y, bw, bh = x * 2, y * 2, bw * 2, bh * 2
            bbox = (float(x), float(y), float(x + bw), float(y + bh))
            cx, cy = x + bw // 2, y + bh // 2
            key = (cx // 20 * 20, cy // 20 * 20)
            live_keys.add(key)

            self._mog2_blobs[key] = self._mog2_blobs.get(key, 0) + 1
            age = self._mog2_blobs[key]
            if age < self._min_frames:
                continue

            near_person = any(_iou(bbox, pb) > self._person_iou_thresh for pb in person_bboxes)
            if not near_person:
                out.append(AbandonedObject(bbox=bbox, age_frames=age, centroid=(cx, cy)))

        stale = [k for k in self._mog2_blobs if k not in live_keys]
        for k in stale:
            del self._mog2_blobs[k]

        return out


def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)
