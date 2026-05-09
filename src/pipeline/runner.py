"""Per-camera pipeline orchestrator.

For each enabled camera we spin up:
  - StreamReader thread (continuously reads latest frame)
  - one inference thread that pulls the latest frame and runs:
      detect -> track -> attach plates by IoU -> OCR plates of unread
      tracks -> signal classify -> rules engine -> persist evidence
      → crowd density + optical flow + stampede risk
      → pose Stage-1 violence heuristics + incident lifecycle
      → loitering zone dwell tracking
      → abandoned object MOG2 detection

The detector/OCR are shared across cameras (they're stateless), so a
single GPU runs N cameras; only the per-camera ingest, tracker, rules,
and state are per-pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from config.settings import settings
from src.attributes.classifier import VehicleAttributeClassifier
from src.crowd.calibration import CameraCalibration
from src.crowd.dense_counter import DenseCrowdCounter
from src.crowd.density import ZoneDensityEstimator
from src.crowd.flow import FlowAnalyzer
from src.crowd.stampede import StampedeDetector
from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.violence.clip_classifier import ViolenceClipClassifier
from src.detection.detector import Detection, Detector
from src.evidence.store import EvidenceStore
from src.ingest.stream import StreamConfig, StreamReader
from src.loitering.detector import AbandonedObjectDetector, LoiteringDetector
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
from src.violence.incident import IncidentManager
from src.violence.pose_stage1 import PoseViolenceDetector
from .annotate import annotate

log = logging.getLogger("pipeline")

# Shared ReID instance — set by PipelineOrchestrator, read by the API.
_shared_reid: Optional["CrossCameraReID"] = None

# Shared crowd state: camera_id → latest state dict (read by API, written by pipeline).
_crowd_state: dict[str, dict] = {}
_crowd_lock = threading.Lock()

# Shared live MJPEG frame bytes: camera_id → latest JPEG bytes (read by API).
_live_frames: dict[str, Optional[bytes]] = {}
_live_frame_lock = threading.Lock()

# Shared mutable rule params — updated by Settings API, read by all pipelines.
_shared_rule_params: Optional[dict] = None
_rule_params_lock = threading.Lock()


def update_crowd_state(camera_id: str, state: dict) -> None:
    with _crowd_lock:
        _crowd_state[camera_id] = state


def get_crowd_state() -> dict[str, dict]:
    with _crowd_lock:
        return dict(_crowd_state)


def update_live_frame(camera_id: str, frame: np.ndarray, quality: int = 72) -> None:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    with _live_frame_lock:
        _live_frames[camera_id] = bytes(buf)


def get_live_frame(camera_id: str) -> Optional[bytes]:
    with _live_frame_lock:
        return _live_frames.get(camera_id)


def get_all_live_cameras() -> list[str]:
    with _live_frame_lock:
        return list(_live_frames.keys())


def get_shared_rule_params() -> Optional[dict]:
    with _rule_params_lock:
        return _shared_rule_params


@dataclass
class CameraSpec:
    id: str
    name: str
    source: str
    fps_cap: int
    stop_line: tuple[tuple[int, int], tuple[int, int]]
    signal_roi: Optional[tuple[int, int, int, int]]
    direction: tuple[float, float]
    meters_per_pixel: float = 0.05
    enabled: bool = True
    loitering_zones: list[dict] = field(default_factory=list)
    crowd_zones: list[dict] = field(default_factory=list)
    homography_pts: Optional[dict] = None


def load_cameras(path: Path) -> list[CameraSpec]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    out: list[CameraSpec] = []
    for entry in data.get("cameras", []):
        if not entry.get("enabled", True):
            continue
        sl = entry["stop_line"]
        hpts = entry.get("homography_pts")
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
                loitering_zones=entry.get("loitering_zones", []),
                crowd_zones=entry.get("crowd_zones", []),
                homography_pts=hpts,
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
        self.rules = RulesEngine(
            camera_id=spec.id,
            geometry=CameraGeometry(
                stop_line_p1=spec.stop_line[0],
                stop_line_p2=spec.stop_line[1],
                direction=spec.direction,
                meters_per_pixel=spec.meters_per_pixel,
            ),
            rule_params=rule_params,
            fps=float(spec.fps_cap),
        )
        self.alert_engine = AlertEngine(rule_params)

        # ── Crowd analytics ──────────────────────────────────────────────
        crowd_cfg = rule_params.get("crowd", {})
        if spec.homography_pts:
            calib = CameraCalibration.from_points(
                spec.homography_pts["image"],
                spec.homography_pts["world"],
            )
        else:
            calib = CameraCalibration.identity()

        self.density_est = ZoneDensityEstimator(
            camera_id=spec.id,
            zones_cfg=spec.crowd_zones,
            calibration=calib,
            danger_density=float(crowd_cfg.get("danger_density", 5.0)),
            warning_density=float(crowd_cfg.get("warning_density", 3.0)),
            danger_count=int(crowd_cfg.get("danger_count", 15)),
            warning_count=int(crowd_cfg.get("warning_count", 8)),
        )
        self.flow_analyzer = FlowAnalyzer(history_len=8)
        self.stampede_det = StampedeDetector(
            danger_density=float(crowd_cfg.get("danger_density", 7.0)),
            warning_threshold=float(crowd_cfg.get("stampede_warning", 0.45)),
            critical_threshold=float(crowd_cfg.get("stampede_critical", 0.70)),
            divergence_min=float(crowd_cfg.get("stampede_divergence_min", 0.35)),
            convergence_min=float(crowd_cfg.get("stampede_convergence_min", 0.30)),
            variance_min=float(crowd_cfg.get("stampede_variance_min", 3.0)),
            counter_flow_min=float(crowd_cfg.get("stampede_counter_flow_min", 0.30)),
        )
        self._snapshot_interval = int(crowd_cfg.get("snapshot_every_frames", 150))
        self._last_snapshot_frame = 0

        # ── Violence detection ───────────────────────────────────────────
        violence_cfg = rule_params.get("violence", {})
        self._violence_enabled = bool(violence_cfg.get("enabled", True))
        self.pose_det = PoseViolenceDetector(
            run_every_n=int(violence_cfg.get("run_every_n_frames", 3)),
            kp_conf=float(violence_cfg.get("kp_confidence", 0.40)),
            proximity_iou=float(violence_cfg.get("proximity_iou", 0.12)),
            wrist_vel_thresh=float(violence_cfg.get("wrist_velocity_threshold", 8.0)),
            device=settings.device,
            weights=settings.pose_weights,
        )
        self.incident_mgr = IncidentManager(spec.id)
        # Stage-2 clip classifier — confirms "suspected" before escalating to "active"
        self._clip_clf = ViolenceClipClassifier(
            device=settings.device,
            threshold=float(violence_cfg.get("clip_confirm_threshold", 0.55)),
            model_name=str(violence_cfg.get("clip_model", "r3d_18")),
        )
        self._clip_confirmation: dict[str, bool] = {}   # camera_id → latest clip result

        # ── Loitering ────────────────────────────────────────────────────
        loiter_cfg = rule_params.get("loitering", {})
        self._loitering_enabled = bool(loiter_cfg.get("enabled", True))
        self.loiter_det = LoiteringDetector(
            camera_id=spec.id,
            zones=spec.loitering_zones,
            fps=float(spec.fps_cap),
            refine_interval_s=float(loiter_cfg.get("cooldown_seconds", 60)),
        )

        # ── Abandoned objects ─────────────────────────────────────────────
        aband_cfg = rule_params.get("abandoned", {})
        self._abandoned_enabled = bool(aband_cfg.get("enabled", True))
        self.abandon_det = AbandonedObjectDetector(
            min_frames=int(aband_cfg.get("min_frames", 300)),
            min_area_px2=int(aband_cfg.get("min_area_px2", 3000)),
            person_iou_thresh=float(aband_cfg.get("person_iou_threshold", 0.15)),
            owner_bind_radius=int(aband_cfg.get("owner_bind_radius_px", 200)),
            owner_separation=int(aband_cfg.get("owner_separation_px", 300)),
            static_frames=int(aband_cfg.get("static_frames", 300)),
        )

        # ── Dense crowd counter ───────────────────────────────────────────
        crowd_cfg2 = rule_params.get("crowd", {})
        self.dense_counter = DenseCrowdCounter(
            dense_trigger_count=int(crowd_cfg2.get("dense_trigger_count", 30)),
            device=settings.device,
        )

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._track_first_seen_frame: dict[int, int] = {}
        self._track_last_seen_frame: dict[int, int] = {}
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
        _drop_warn_accumulator = 0
        from src.common.metrics import frames_total, frame_drops_total
        while not self._stop.is_set():
            packet = self.reader.read()
            if packet is None:
                time.sleep(0.005)
                continue
            frame_idx, ts, frame = packet
            if frame_idx == last_frame_idx:
                time.sleep(0.002)
                continue

            skip_count = frame_idx - last_frame_idx - 1
            if skip_count > 0:
                frame_drops_total.labels(camera_id=self.spec.id).inc(skip_count)
                _drop_warn_accumulator += skip_count
                # Only warn once per 60-frame burst to avoid log spam
                if _drop_warn_accumulator >= 60:
                    log.warning("[%s] inference lagging — dropped %d frames in last burst", self.spec.id, _drop_warn_accumulator)
                    _drop_warn_accumulator = 0
            else:
                _drop_warn_accumulator = 0

            last_frame_idx = frame_idx
            frames_total.labels(camera_id=self.spec.id).inc()

            # Push every raw frame to MJPEG buffer immediately so the UI stays live.
            # _annotate_crowd_on_live_frame will overwrite with overlays once available.
            update_live_frame(self.spec.id, frame)

            try:
                self._process_frame(frame_idx, ts, frame)
            except Exception:
                log.exception("[%s] frame %d failed", self.spec.id, frame_idx)

    def _process_frame(self, frame_idx: int, ts: float, frame: np.ndarray) -> None:
        bundle = self.detector(frame, frame_idx, ts)
        tracked = self.tracker.update(bundle)

        # Attach plates → tracks by IoU, run OCR, propagate to rules engine.
        plate_dets = bundle.of(ObjectClass.LICENSE_PLATE)
        person_bboxes: list[tuple] = []
        luggage_dets: list[tuple] = []

        # Collect luggage detections for owner-association abandoned object detection
        for ld in bundle.of(ObjectClass.LUGGAGE):
            luggage_dets.append((ld.xyxy[0], ld.xyxy[1], ld.xyxy[2], ld.xyxy[3]))

        # Feed frame to clip violence classifier buffer every frame
        if self._violence_enabled and self._clip_clf.available:
            self._clip_clf.submit_frame(self.spec.id, frame)

        for td in tracked:
            cls = td.detection.cls
            if cls == ObjectClass.PERSON:
                xyxy = td.detection.xyxy
                person_bboxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))

            if cls not in VEHICLE_CLASSES:
                continue
            self._track_first_seen_frame.setdefault(td.track_id, frame_idx)
            self._track_last_seen_frame[td.track_id] = frame_idx

            # --- ReID + attributes ---
            if self.reid is not None:
                plate_text_for_reid, _ = self.rules.best_plate(td.track_id)
                attrs = self._track_attrs.get(td.track_id)
                color = attrs[1] if attrs else None
                vtype = attrs[2] if attrs else None

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

                if propagated_plate and not plate_text_for_reid:
                    self.rules.attach_plate_read(td.track_id, propagated_plate, 0.60)

                self._track_attrs[td.track_id] = (global_id, color, vtype)

                if match_sim > 0:
                    self._persist_reid_subject(global_id, match_sim, td, color, vtype)

            # --- plate OCR ---
            plate = self._best_plate_for_vehicle(td.detection, plate_dets)
            if plate is None:
                continue
            # fast-alpr pre-reads the text — skip OCR entirely
            if plate.text and plate.text_conf is not None and plate.text_conf >= 0.30:
                normalized = normalize_indian_plate(plate.text)
                if normalized:
                    self.rules.attach_plate_read(td.track_id, normalized, plate.text_conf)
                continue
            # Legacy: crop + PlateOCR
            crop = _crop(frame, plate.xyxy)
            read = self.ocr.read(crop)
            if not read:
                continue
            normalized = normalize_indian_plate(read.text)
            self.rules.attach_plate_read(td.track_id, normalized, read.confidence)

        signal_state = self.signal_clf.classify(frame)
        raw_events = self.rules.step(tracked, bundle, signal_state)
        raw_events += self._check_plate_unreadable(tracked, bundle, frame_idx, ts)

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

        # ── Step 3: Crowd analytics ──────────────────────────────────────
        self._run_crowd_analytics(frame, frame_idx, person_bboxes)

        # ── Step 3: Violence detection ───────────────────────────────────
        if self._violence_enabled:
            self._run_violence_detection(frame, frame_idx, person_bboxes)

        # ── Step 3: Loitering detection ──────────────────────────────────
        if self._loitering_enabled and person_bboxes:
            self._run_loitering(tracked, frame_idx)

        # ── Step 3: Abandoned object detection ───────────────────────────
        if self._abandoned_enabled:
            self._run_abandoned(frame, frame_idx, person_bboxes, luggage_dets)

    # ----------------------------------------------------- Step 3 helpers

    def _run_crowd_analytics(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_bboxes: list[tuple],
    ) -> None:
        crowd = self.density_est.update(person_bboxes)
        flow = self.flow_analyzer.update(frame)
        flow_smooth = self.flow_analyzer.smoothed()
        stampede = self.stampede_det.assess(crowd, flow_smooth)

        # Dense-crowd correction: YOLO under-counts above ~30 persons/frame
        adjusted_count = self.dense_counter.get_adjusted_count(
            crowd.total_count, person_bboxes, frame
        )

        state = {
            "camera_id": self.spec.id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_count": adjusted_count,
            "yolo_raw_count": crowd.total_count,
            "dense_mode": self.dense_counter.mode if adjusted_count != crowd.total_count else "yolo",
            "max_density": crowd.max_density,
            "risk_level": crowd.risk_level,
            "stampede_score": stampede.score,
            "stampede_level": stampede.level,
            "stampede_flags": stampede.flags,
            "flow": {
                "magnitude": round(flow.magnitude, 2),
                "variance": round(flow.variance, 2),
                "divergence": round(flow.divergence, 3),
                "convergence": round(flow.convergence, 3),
                "counter_flow": round(flow.counter_flow, 3),
            },
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "label": z.label,
                    "count": z.person_count,
                    "density": round(z.density, 2),
                }
                for z in crowd.zones
            ],
        }
        update_crowd_state(self.spec.id, state)

        # Annotate live frame with crowd info (shown in MJPEG feed)
        self._annotate_crowd_on_live_frame(frame, crowd, stampede)

        # Write to DB on warning/critical or every snapshot_interval frames
        crowd_needs_snapshot = (
            stampede.level != "ok"
            or crowd.risk_level != "ok"
            or (frame_idx - self._last_snapshot_frame) >= self._snapshot_interval
        )
        if crowd_needs_snapshot and crowd.total_count > 0:
            self._last_snapshot_frame = frame_idx
            self._persist_crowd_snapshot(crowd, stampede)

        # Push critical stampede event via WS
        if stampede.level == "critical":
            try:
                from src.api.server import push_violation_event
                push_violation_event({
                    "type": "stampede",
                    "camera_id": self.spec.id,
                    "score": stampede.score,
                    "level": stampede.level,
                    "flags": stampede.flags,
                    "total_count": crowd.total_count,
                })
            except Exception:
                pass

    def _on_clip_result(self, result) -> None:
        """Callback from async clip classifier."""
        self._clip_confirmation[result.camera_id] = result.confirmed
        log.debug(
            "[%s] clip result: score=%.3f confirmed=%s label=%s",
            result.camera_id, result.score, result.confirmed, result.label,
        )

    def _run_violence_detection(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_bboxes: list[tuple],
    ) -> None:
        result = self.pose_det.update(frame, frame_idx, person_bboxes)

        # Stage-2 gate: when pose heuristic fires "suspected", request a clip check
        if result.active and self._clip_clf.available:
            self._clip_clf.request_inference(self.spec.id, self._on_clip_result)
            # If clip classifier hasn't confirmed yet, hold at suspected
            if not self._clip_confirmation.get(self.spec.id, False):
                # Force only "suspected" until clip confirms
                from dataclasses import replace as dc_replace
                result = dc_replace(result, active=False)

        changed = self.incident_mgr.update(result)
        if changed is not None:
            self._persist_violence_incident(changed)
            if changed.status in ("active", "suspected"):
                try:
                    from src.api.server import push_violation_event
                    push_violation_event({
                        "type": "incident",
                        "itype": "violence",
                        "camera_id": self.spec.id,
                        "status": changed.status,
                        "score": changed.score,
                        "flags": changed.flags,
                        "db_id": changed.db_id,
                    })
                except Exception:
                    pass

    def _run_loitering(
        self,
        tracked: list[TrackedDetection],
        frame_idx: int,
    ) -> None:
        persons = [
            (td.track_id, *td.detection.xyxy)
            for td in tracked
            if td.detection.cls == ObjectClass.PERSON
        ]
        events = self.loiter_det.update(persons)
        for ev in events:
            self._persist_loiter_incident(ev)
            try:
                from src.api.server import push_violation_event
                push_violation_event({
                    "type": "incident",
                    "itype": "loitering",
                    "camera_id": ev.camera_id,
                    "track_id": ev.track_id,
                    "zone_id": ev.zone_id,
                    "zone_label": ev.zone_label,
                    "dwell_seconds": ev.dwell_seconds,
                    "status": "active",
                })
            except Exception:
                pass

    def _run_abandoned(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_bboxes: list[tuple],
        luggage_dets: Optional[list[tuple]] = None,
    ) -> None:
        objects = self.abandon_det.update(
            frame,
            person_bboxes,
            object_detections=luggage_dets if luggage_dets else None,
        )
        for obj in objects:
            # Only push/persist every ~30 frames per object centroid
            if frame_idx % 30 == 0:
                self._persist_abandoned_incident(obj, frame_idx)
                try:
                    from src.api.server import push_violation_event
                    push_violation_event({
                        "type": "incident",
                        "itype": "abandoned",
                        "camera_id": self.spec.id,
                        "age_frames": obj.age_frames,
                        "centroid": obj.centroid,
                        "status": "active",
                    })
                except Exception:
                    pass

    # ----------------------------------------------------- live frame annotator

    def _annotate_crowd_on_live_frame(self, frame: np.ndarray, crowd, stampede) -> None:
        """Overlay crowd stats on the live MJPEG frame buffer."""
        try:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            # Semi-transparent dark strip at top
            cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            level_color = {
                "ok": (80, 200, 80),
                "warning": (40, 170, 210),
                "critical": (50, 50, 220),
            }.get(stampede.level, (200, 200, 200))

            txt = (
                f"{self.spec.id}  |  persons:{crowd.total_count}"
                f"  density:{crowd.max_density:.1f}p/m²"
                f"  risk:{stampede.level.upper()}"
            )
            cv2.putText(frame, txt, (8, 18), font, 0.45, level_color, 1, cv2.LINE_AA)

            # Write annotated frame to live buffer
            update_live_frame(self.spec.id, frame, quality=65)
        except Exception:
            pass

    # ----------------------------------------------------- DB persisters

    def _persist_crowd_snapshot(self, crowd, stampede) -> None:
        try:
            from src.common.db import session_scope
            from src.evidence.models import CrowdSnapshot
            with session_scope() as s:
                s.add(CrowdSnapshot(
                    camera_id=self.spec.id,
                    timestamp=datetime.utcnow(),
                    total_count=crowd.total_count,
                    max_density=crowd.max_density,
                    stampede_score=stampede.score,
                    stampede_level=stampede.level,
                    stampede_flags=stampede.flags,
                    zones_json=[
                        {"id": z.zone_id, "label": z.label, "count": z.person_count, "density": round(z.density, 2)}
                        for z in crowd.zones
                    ],
                ))
        except Exception:
            log.debug("crowd snapshot persist failed", exc_info=True)

    def _persist_violence_incident(self, inc) -> None:
        try:
            from src.common.db import session_scope
            from src.evidence.models import Incident
            with session_scope() as s:
                if inc.db_id is None:
                    row = Incident(
                        camera_id=inc.camera_id,
                        itype="violence",
                        started_at=inc.started_at,
                        status=inc.status,
                        score=inc.score,
                        flags=inc.flags,
                    )
                    s.add(row)
                    s.flush()
                    inc.db_id = row.id
                else:
                    row = s.get(Incident, inc.db_id)
                    if row:
                        row.status = inc.status
                        row.score = inc.score
                        row.flags = inc.flags
                        if inc.resolved_at:
                            row.resolved_at = inc.resolved_at
        except Exception:
            log.debug("violence incident persist failed", exc_info=True)

    def _persist_loiter_incident(self, ev) -> None:
        try:
            from src.common.db import session_scope
            from src.evidence.models import Incident
            with session_scope() as s:
                s.add(Incident(
                    camera_id=ev.camera_id,
                    itype="loitering",
                    started_at=datetime.utcnow(),
                    status="active",
                    score=min(ev.dwell_seconds / 120.0, 1.0),
                    track_id=ev.track_id,
                    zone_id=ev.zone_id,
                    extras={"dwell_seconds": ev.dwell_seconds, "zone_label": ev.zone_label},
                ))
        except Exception:
            log.debug("loiter incident persist failed", exc_info=True)

    def _persist_abandoned_incident(self, obj, frame_idx: int) -> None:
        try:
            from src.common.db import session_scope
            from src.evidence.models import Incident
            with session_scope() as s:
                s.add(Incident(
                    camera_id=self.spec.id,
                    itype="abandoned",
                    started_at=datetime.utcnow(),
                    status="active",
                    score=min(obj.age_frames / 600.0, 1.0),
                    extras={"age_frames": obj.age_frames, "centroid": list(obj.centroid), "bbox": list(obj.bbox)},
                ))
        except Exception:
            log.debug("abandoned incident persist failed", exc_info=True)

    # ----------------------------------------------------- traffic helpers

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
        try:
            from src.common.db import session_scope
            from src.evidence.models import ReidSubject
            plate_text, _ = self.rules.best_plate(td.track_id)
            now = datetime.utcnow()
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
                        row.camera_ids = cameras + [self.spec.id]
        except Exception:
            pass


def _crop(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------- top level

class PipelineOrchestrator:
    def __init__(self):
        import os
        import torch
        # Use all available CPU threads for data-parallel operations (OCR, pre/post-process).
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(max(2, cpu_count // 2))
        log.info("CPU parallelism: %d inference threads, %d interop threads", cpu_count, max(2, cpu_count // 2))

        if torch.cuda.is_available():
            # Allow PyTorch to overlap CPU↔GPU transfers with compute
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            log.info("CUDA: %s | TF32+cuDNN benchmark enabled", torch.cuda.get_device_name(0))

        from src.common.db import engine
        from src.evidence.models import Base
        Base.metadata.create_all(bind=engine())

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
            use_fast_alpr=settings.use_fast_alpr,
        )
        self.ocr = PlateOCR(lang=settings.ocr_lang, use_gpu=settings.device.startswith("cuda"))
        self.store = EvidenceStore()
        self.rule_params = load_rules(settings.rules_yaml)
        self.cameras = load_cameras(settings.cameras_yaml)

        # Expose rule_params so Settings API can mutate + persist it
        global _shared_rule_params
        with _rule_params_lock:
            _shared_rule_params = self.rule_params
        self.pipelines: list[CameraPipeline] = []

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
