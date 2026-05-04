"""Per-camera pipeline orchestrator.

For each enabled camera we spin up:
  - StreamReader thread (continuously reads latest frame)
  - one inference thread that pulls the latest frame and runs:
      detect -> track -> attach plates by IoU -> OCR plates of unread
      tracks -> signal classify -> rules engine -> persist evidence

The detector/OCR are shared across cameras (they're stateless), so a
single GPU runs N cameras; only the per-camera ingest, tracker, rules,
and state are per-pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from config.settings import settings
from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.detection.detector import Detection, Detector
from src.evidence.store import EvidenceStore
from src.ingest.stream import StreamConfig, StreamReader
from src.ocr.plate_normalize import normalize_indian_plate
from src.ocr.plate_ocr import PlateOCR
from src.rules.engine import CameraGeometry, RulesEngine
from src.rules.geometry import contains
from src.rules.types import ViolationCode, ViolationEvent
from src.signal.classifier import SignalClassifier, SignalState
from src.tracking.tracker import Tracker, TrackedDetection
from .annotate import annotate

log = logging.getLogger("pipeline")


@dataclass
class CameraSpec:
    id: str
    name: str
    source: str
    fps_cap: int
    stop_line: tuple[tuple[int, int], tuple[int, int]]
    signal_roi: Optional[tuple[int, int, int, int]]
    direction: tuple[float, float]
    enabled: bool = True


def load_cameras(path: Path) -> list[CameraSpec]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    out: list[CameraSpec] = []
    for entry in data.get("cameras", []):
        if not entry.get("enabled", True):
            continue
        sl = entry["stop_line"]
        out.append(
            CameraSpec(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                source=entry["source"],
                fps_cap=int(entry.get("fps_cap", 15)),
                stop_line=((int(sl[0][0]), int(sl[0][1])), (int(sl[1][0]), int(sl[1][1]))),
                signal_roi=tuple(entry["signal_roi"]) if entry.get("signal_roi") else None,
                direction=tuple(entry.get("direction", [0.0, -1.0])),
                enabled=True,
            )
        )
    return out


def load_rules(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class CameraPipeline:
    def __init__(
        self,
        spec: CameraSpec,
        detector: Detector,
        ocr: PlateOCR,
        store: EvidenceStore,
        rule_params: dict,
    ):
        self.spec = spec
        self.detector = detector
        self.ocr = ocr
        self.store = store
        self.rule_params = rule_params

        self.reader = StreamReader(StreamConfig(camera_id=spec.id, source=spec.source, fps_cap=spec.fps_cap))
        self.tracker = Tracker(frame_rate=spec.fps_cap)
        self.signal_clf = SignalClassifier(spec.signal_roi)
        self.rules = RulesEngine(
            camera_id=spec.id,
            geometry=CameraGeometry(
                stop_line_p1=spec.stop_line[0], stop_line_p2=spec.stop_line[1], direction=spec.direction
            ),
            rule_params=rule_params,
        )

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._track_first_seen_frame: dict[int, int] = {}
        self._track_last_seen_frame: dict[int, int] = {}

    def start(self) -> None:
        self.reader.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"pipeline-{self.spec.id}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4.0)
        self.reader.stop()

    # ----------------------------------------------------------- main loop

    def _loop(self) -> None:
        last_frame_idx = -1
        while not self._stop.is_set():
            packet = self.reader.read()
            if packet is None:
                time.sleep(0.01)
                continue
            frame_idx, ts, frame = packet
            if frame_idx == last_frame_idx:
                time.sleep(0.005)
                continue
            last_frame_idx = frame_idx

            try:
                self._process_frame(frame_idx, ts, frame)
            except Exception:
                log.exception("[%s] frame %d failed", self.spec.id, frame_idx)

    def _process_frame(self, frame_idx: int, ts: float, frame: np.ndarray) -> None:
        bundle = self.detector(frame, frame_idx, ts)
        tracked = self.tracker.update(bundle)

        # Attach plates → tracks by spatial containment, run OCR for new reads.
        plate_dets = bundle.of(ObjectClass.LICENSE_PLATE)
        for td in tracked:
            if td.detection.cls not in VEHICLE_CLASSES:
                continue
            self._track_first_seen_frame.setdefault(td.track_id, frame_idx)
            self._track_last_seen_frame[td.track_id] = frame_idx

            plate = self._best_plate_for_vehicle(td.detection, plate_dets)
            if plate is None:
                continue
            crop = _crop(frame, plate.xyxy)
            read = self.ocr.read(crop)
            if not read:
                continue
            normalized = normalize_indian_plate(read.text)
            self.rules.attach_plate_read(td.track_id, normalized, read.confidence)

        signal_state = self.signal_clf.classify(frame)
        events = self.rules.step(tracked, bundle, signal_state)

        # Detect plate-unreadable for tracks that left the frame this step.
        events += self._check_plate_unreadable(tracked, bundle, frame_idx, ts)

        if events:
            annotated = annotate(frame, bundle, tracked, stop_line=self.spec.stop_line, signal_state=signal_state.value)
            for ev in events:
                plate_crop_img = self._plate_crop_for_event(ev, bundle, frame)
                self.store.save(ev, frame=frame, annotated=annotated, plate_crop=plate_crop_img)

    # ----------------------------------------------------- helpers

    def _best_plate_for_vehicle(self, vehicle: Detection, plates: list[Detection]) -> Optional[Detection]:
        best = None
        best_score = 0.0
        for p in plates:
            score = contains(vehicle.xyxy, p.xyxy)
            if score > best_score:
                best_score = score
                best = p
        return best if best_score >= 0.6 else None

    def _plate_crop_for_event(self, ev: ViolationEvent, bundle, frame: np.ndarray) -> Optional[np.ndarray]:
        if not ev.bbox:
            return None
        # find plate inside the offender's vehicle box
        for p in bundle.of(ObjectClass.LICENSE_PLATE):
            if contains(ev.bbox, p.xyxy) >= 0.6:
                return _crop(frame, p.xyxy)
        return None

    def _check_plate_unreadable(
        self,
        tracked: list[TrackedDetection],
        bundle,
        frame_idx: int,
        ts: float,
    ) -> list[ViolationEvent]:
        cfg = self.rule_params.get("plate_unreadable", {})
        if not cfg.get("enabled"):
            return []
        out: list[ViolationEvent] = []
        # Identify tracks we just lost: present last frame, absent this one.
        active = {td.track_id for td in tracked}
        lost = [tid for tid, last in self._track_last_seen_frame.items()
                if last < frame_idx - 5 and tid not in active]
        for tid in lost:
            first = self._track_first_seen_frame.get(tid, frame_idx)
            if frame_idx - first < cfg.get("min_track_frames", 25):
                self._forget_track(tid)
                continue
            text, conf = self.rules.best_plate(tid)
            if text and conf >= settings.ocr_auto_accept:
                self._forget_track(tid)
                continue
            out.append(ViolationEvent(
                code=ViolationCode.PLATE_UNREADABLE,
                camera_id=self.spec.id,
                track_id=tid,
                timestamp=ts,
                frame_idx=frame_idx,
                confidence=1.0 - max(conf, 0.0),
                plate_text=text,
                plate_ocr_confidence=conf,
                bbox=None,
            ))
            self._forget_track(tid)
        return out

    def _forget_track(self, tid: int) -> None:
        self._track_first_seen_frame.pop(tid, None)
        self._track_last_seen_frame.pop(tid, None)


def _crop(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------- top level

class PipelineOrchestrator:
    def __init__(self):
        from src.common.db import engine
        from src.evidence.models import Base
        Base.metadata.create_all(bind=engine())

        self.detector = Detector(
            general_weights=settings.detector_weights,
            helmet_weights=settings.helmet_weights,
            plate_weights=settings.plate_weights,
            device=settings.device,
            conf=settings.det_conf,
        )
        self.ocr = PlateOCR(lang=settings.ocr_lang, use_gpu=settings.device.startswith("cuda"))
        self.store = EvidenceStore()
        self.rule_params = load_rules(settings.rules_yaml)
        self.cameras = load_cameras(settings.cameras_yaml)
        self.pipelines: list[CameraPipeline] = []

    def run(self) -> None:
        if not self.cameras:
            log.warning("No enabled cameras in %s", settings.cameras_yaml)
            return
        for spec in self.cameras:
            log.info("Starting pipeline for %s (%s)", spec.id, spec.name)
            p = CameraPipeline(spec, self.detector, self.ocr, self.store, self.rule_params)
            p.start()
            self.pipelines.append(p)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Shutting down")
            for p in self.pipelines:
                p.stop()
