"""Multi-object tracking via ByteTrack (through the `supervision` library).

ByteTrack associates detections across frames so we can give each
vehicle/rider a stable integer ID. The rules engine fires only when the
*same* track exhibits a violation across multiple frames — that's how we
avoid false positives from single-frame detector hiccups.

We track only vehicle/person classes; helmet and plate boxes are not
tracked individually — they're attached to the parent vehicle by spatial
overlap inside the rules engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import supervision as sv

from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.detection.detector import Detection, DetectionBundle

TRACKABLE = VEHICLE_CLASSES | {ObjectClass.PERSON}


@dataclass
class TrackedDetection:
    track_id: int
    detection: Detection


class Tracker:
    def __init__(self, frame_rate: int = 15):
        # ByteTrack defaults are reasonable for traffic. We expose
        # frame_rate so its internal time constants stay correct when
        # the pipeline is FPS-capped.
        self._tracker = sv.ByteTrack(frame_rate=frame_rate)

    def update(self, bundle: DetectionBundle) -> list[TrackedDetection]:
        # supervision wants Detections in its own format
        trackable_dets = [d for d in bundle.detections if d.cls in TRACKABLE]
        if not trackable_dets:
            return []

        xyxy = np.array([d.xyxy for d in trackable_dets], dtype=np.float32)
        conf = np.array([d.conf for d in trackable_dets], dtype=np.float32)
        # supervision needs class_id ints; we keep our own enum mapping
        cls_idx = np.array([list(ObjectClass).index(d.cls) for d in trackable_dets], dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_idx)
        tracked = self._tracker.update_with_detections(sv_dets)

        out: list[TrackedDetection] = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            if tid < 0:
                continue
            cls = list(ObjectClass)[int(tracked.class_id[i])]
            box = tuple(map(float, tracked.xyxy[i]))
            det = Detection(cls=cls, conf=float(tracked.confidence[i]), xyxy=box)
            out.append(TrackedDetection(track_id=tid, detection=det))
        return out
