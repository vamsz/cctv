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
from src.attributes.classifier import VehicleAttributeClassifier
from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.detection.detector import Detection, Detector
from src.evidence.store import EvidenceStore
from src.ingest.stream import StreamConfig, StreamReader
from src.ocr.plate_normalize import normalize_indian_plate
from src.ocr.plate_ocr import PlateOCR
from src.reid.matcher import CrossCameraReID
from src.rules.alert_engine import AlertEngine
from src.rules.engine import CameraGeometry, RulesEngine
from src.rules.geometry import contains
from src.rules.types import ViolationCode, ViolationEvent
from src.rules import watchlist as wl
from src.signal.classifier import SignalClassifier, SignalState
from src.tracking.tracker import Tracker, TrackedDetection
from .annotate import annotate

log = logging.getLogger("pipeline")

# Shared ReID instance — set by PipelineOrchestrator, read by the API.
_shared_reid: Optional["CrossCameraReID"] = None


@dataclass
class CameraSpec:
    id: str
    name: str
    source: str
    fps_cap: int
    stop_line: tuple[tuple[int, int], tuple[int, int]]
    signal_roi: Optional[tuple[int, int, int, int]]
    direction: tuple[float, float]
    meters_per_pixel: float = 0.05    # calibrated; used for speed estimation
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
                meters_per_pixel=float(entry.get("meters_per_pixel", 0.05)),
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
        reid: Optional[CrossCameraReID] = None,
        attr_clf: Optional[VehicleAttributeClassifier] = None,
    ):
        self.spec = spec
        self.detector = detector
        self.ocr = ocr
        self.store = store
        self.rule_params = rule_params
        self.reid = reid
        self.attr_clf = attr_clf

        self.reader = StreamReader(StreamConfig(camera_id=spec.id, source=spec.source, fps_cap=spec.fps_cap))
        self.tracker = Tracker(frame_rate=spec.fps_cap)
        self.signal_clf = SignalClassifier(spec.signal_roi)
        meters_per_pixel = spec.meters_per_pixel if hasattr(spec, "meters_per_pixel") else 0.05
        self.rules = RulesEngine(
            camera_id=spec.id,
            geometry=CameraGeometry(
                stop_line_p1=spec.stop_line[0],
                stop_line_p2=spec.stop_line[1],
                direction=spec.direction,
                meters_per_pixel=meters_per_pixel,
            ),
            rule_params=rule_params,
            fps=float(spec.fps_cap),
        )
        self.alert_engine = AlertEngine(rule_params)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._track_first_seen_frame: dict[int, int] = {}
        self._track_last_seen_frame: dict[int, int] = {}
        # cache: track_id -> (global_id, color, type)
        self._track_attrs: dict[int, tuple] = {}

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

        # Attach plates → tracks by IoU, run OCR, propagate to rules engine.
        plate_dets = bundle.of(ObjectClass.LICENSE_PLATE)
        for td in tracked:
            if td.detection.cls not in VEHICLE_CLASSES:
                continue
            self._track_first_seen_frame.setdefault(td.track_id, frame_idx)
            self._track_last_seen_frame[td.track_id] = frame_idx

            # --- ReID + attributes (every N frames per track) ---
            if self.reid is not None:
                plate_text_for_reid, _ = self.rules.best_plate(td.track_id)
                attrs = self._track_attrs.get(td.track_id)
                color = attrs[1] if attrs else None
                vtype = attrs[2] if attrs else None

                # Run attribute classifier if we don't have it yet
                if not color and self.attr_clf:
                    crop = _crop(frame, td.detection.xyxy)
                    result = self.attr_clf.classify(crop, td.detection.cls)
                    color = result["color"]
                    vtype = result["type"]

                global_id, match_sim, propagated_plate = self.reid.process(
                    frame=frame,
                    xyxy=td.detection.xyxy,
                    camera_id=self.spec.id,
                    track_id=td.track_id,
                    frame_idx=frame_idx,
                    plate_text=plate_text_for_reid,
                    vehicle_color=color,
                    vehicle_type=vtype,
                )

                # If a cross-camera match propagated a plate we didn't have, attach it
                if propagated_plate and not plate_text_for_reid:
                    self.rules.attach_plate_read(td.track_id, propagated_plate, 0.60)

                self._track_attrs[td.track_id] = (global_id, color, vtype)

                if match_sim > 0:
                    self._persist_reid_subject(global_id, match_sim, td, color, vtype)

            # --- plate OCR ---
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
        raw_events = self.rules.step(tracked, bundle, signal_state)

        # Plate-unreadable for tracks that left the frame this step.
        raw_events += self._check_plate_unreadable(tracked, bundle, frame_idx, ts)

        # Attach ReID global_id + attributes to each event's extras
        for ev in raw_events:
            attrs = self._track_attrs.get(ev.track_id)
            if attrs:
                global_id, color, vtype = attrs
                if global_id:
                    ev.extras["reid_global_id"] = global_id
                if color:
                    ev.extras["vehicle_color"] = color
                if vtype:
                    ev.extras["vehicle_type"] = vtype

        # Global kill-switch + priority-based cooldowns.
        events = self.alert_engine.filter(raw_events)

        if events:
            annotated = annotate(
                frame, bundle, tracked,
                stop_line=self.spec.stop_line,
                signal_state=signal_state.value,
                violations=events,
            )
            for ev in events:
                plate_crop_img = self._plate_crop_for_event(ev, bundle, frame)
                row_id = self.store.save(ev, frame=frame, annotated=annotated, plate_crop=plate_crop_img)
                try:
                    from src.api.server import push_violation_event
                    push_violation_event({
                        "type": "violation",
                        "id": row_id,
                        "code": ev.code.value,
                        "camera_id": ev.camera_id,
                        "track_id": ev.track_id,
                        "plate_text": ev.plate_text,
                        "confidence": round(ev.confidence, 3),
                        "extras": ev.extras,
                    })
                except Exception:
                    pass

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
        self._track_attrs.pop(tid, None)
        if self.reid:
            self.reid.forget_track(self.spec.id, tid)

    def _persist_reid_subject(
        self,
        global_id: str,
        match_sim: float,
        td: TrackedDetection,
        color: Optional[str],
        vtype: Optional[str],
    ) -> None:
        """Write/update the reid_subjects DB row (best-effort, non-blocking)."""
        try:
            from datetime import datetime as _dt
            from sqlalchemy import select
            from src.common.db import session_scope
            from src.evidence.models import ReidSubject
            plate_text, _ = self.rules.best_plate(td.track_id)
            now = _dt.utcnow()
            with session_scope() as s:
                row = s.get(ReidSubject, global_id)
                if row is None:
                    row = ReidSubject(
                        global_id=global_id,
                        first_seen_at=now,
                        last_seen_at=now,
                        camera_ids=[self.spec.id],
                        plate_text=plate_text,
                        vehicle_color=color,
                        vehicle_type=vtype,
                        match_count=1,
                    )
                    s.add(row)
                else:
                    row.last_seen_at = now
                    row.match_count = (row.match_count or 0) + 1
                    if plate_text and not row.plate_text:
                        row.plate_text = plate_text
                    cameras = row.camera_ids or []
                    if self.spec.id not in cameras:
                        cameras = cameras + [self.spec.id]
                        row.camera_ids = cameras
        except Exception:
            pass


def _crop(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------- top level

class PipelineOrchestrator:
    def __init__(self):
        from src.common.db import engine
        from src.evidence.models import Base
        Base.metadata.create_all(bind=engine())

        # Watchlist
        wl.load_from_csv(Path("data/watchlist_seed.csv"))
        wl.load_from_db()
        wl.start_background_reload()

        self.detector = Detector(
            general_weights=settings.detector_weights,
            helmet_weights=settings.helmet_weights,
            plate_weights=settings.plate_weights,
            device=settings.device,
            conf=settings.det_conf,
            use_sahi_plate=settings.sahi_plate_enabled,
        )
        self.ocr = PlateOCR(lang=settings.ocr_lang, use_gpu=settings.device.startswith("cuda"))
        self.store = EvidenceStore()
        self.rule_params = load_rules(settings.rules_yaml)
        self.cameras = load_cameras(settings.cameras_yaml)
        self.pipelines: list[CameraPipeline] = []

        # Shared cross-camera ReID (one embedder + one store shared across all cameras)
        global _shared_reid
        self.reid: Optional[CrossCameraReID] = None
        if settings.reid_enabled:
            log.info("ReID enabled — loading MobileNetV3-Small embedder on %s", settings.device)
            self.reid = CrossCameraReID(
                device=settings.device,
                cosine_threshold=settings.reid_threshold,
                max_age_seconds=settings.reid_max_age_seconds,
                extract_every_n_frames=settings.reid_extract_every_n,
            )
            _shared_reid = self.reid

        # Shared vehicle attribute classifier (fast, CPU, no GPU needed)
        self.attr_clf = VehicleAttributeClassifier()

    def run(self) -> None:
        if not self.cameras:
            log.warning("No enabled cameras in %s", settings.cameras_yaml)
            return
        for spec in self.cameras:
            log.info("Starting pipeline for %s (%s)", spec.id, spec.name)
            p = CameraPipeline(
                spec, self.detector, self.ocr, self.store, self.rule_params,
                reid=self.reid,
                attr_clf=self.attr_clf,
            )
            p.start()
            self.pipelines.append(p)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Shutting down")
            for p in self.pipelines:
                p.stop()
