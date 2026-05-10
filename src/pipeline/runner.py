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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from config.settings import settings
from src.attributes.classifier import VehicleAttributeClassifier
from src.crowd.calibration import CameraCalibration
from src.common.quality import default_gate as _default_quality_gate
from src.crowd.dense_counter import DenseCrowdCounter
from src.crowd.density import ZoneDensityEstimator
from src.crowd.flow import FlowAnalyzer
from src.crowd.fruin import FruinClassifier, SafetyMonitor
from src.crowd.stampede import StampedeDetector
from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.violence.clip_classifier import ViolenceClipClassifier
from src.violence.frame_classifier import FrameViolenceClassifier
from src.detection.detector import Detection, Detector
from src.evidence.store import EvidenceStore
from src.ingest.stream import StreamConfig, StreamReader
from src.loitering.detector import AbandonedObjectDetector, LoiteringDetector
from src.ocr.plate_normalize import normalize_plate
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
        face_detector=None,
        face_embedder=None,
        police_db=None,
    ):
        self.spec = spec
        self.detector = detector
        self.ocr = ocr
        self.store = store
        self.rule_params = rule_params
        self.reid = reid
        self.attr_clf = attr_clf
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.police_db = police_db
        # Per-incident, per-track face capture state.
        # Each incident gets an IncidentFaceBuffer; each unique track_id
        # within that incident gets up to `per_track_capacity` slots.
        from src.face.best_view import IncidentFaceBuffer
        self._face_buffers: dict[int, IncidentFaceBuffer] = {}
        self._face_per_track_capacity = 3
        self._face_total_capacity = settings.face_capture_max_per_incident
        self._last_face_scan_frame: int = -10_000
        # (incident_id, track_id_key) -> list of FaceCapture row ids
        # Lets us UPDATE an existing row when a better view of the SAME
        # person comes in, instead of INSERTing endlessly.
        self._face_track_db_ids: dict[tuple, list[int]] = {}

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
        # Fruin LoS + surge/stagnation/counter-flow alerts
        self.safety_monitor = SafetyMonitor(
            surge_threshold=float(crowd_cfg.get("surge_threshold", 1.0)),
            surge_window=float(crowd_cfg.get("surge_window_seconds", 30.0)),
            counter_flow_threshold=float(crowd_cfg.get("counter_flow_alert", 0.6)),
            stagnation_threshold=float(crowd_cfg.get("stagnation_alert", 0.5)),
            alert_cooldown=float(crowd_cfg.get("alert_cooldown_seconds", 30.0)),
        )
        # Frame quality gate for VideoMAE — skip blurry / black frames
        self.quality_gate = _default_quality_gate()

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
        # Stage-2 clip classifier — primary trigger, polled periodically
        self._clip_clf = ViolenceClipClassifier(
            device=settings.device,
            threshold=float(violence_cfg.get("clip_confirm_threshold", 0.55)),
            model_name=str(violence_cfg.get("clip_model", "r3d_18")),
        )
        self._clip_confirmed: bool = False
        self._clip_score: float = 0.0
        self._clip_label: str = ""
        self._clip_last_update_frame: int = -10_000
        # Frequency at which we *poll* the clip classifier when nothing else
        # has triggered it. ~2 s at 30 fps → user sees a result fast.
        self._clip_poll_interval = int(violence_cfg.get("clip_poll_interval_frames", 60))
        self._last_clip_request_frame: int = -10_000
        # If no positive clip result for this many frames, treat the camera
        # as quiet (lets IncidentManager auto-resolve).
        self._clip_stale_after = int(violence_cfg.get("clip_stale_after_frames", 90))
        # Latest pipeline frame we processed — used by the async clip callback.
        self._latest_inference_frame_idx: int = 0

        # CLIP zero-shot per-frame classifier — production frame-level
        # signal. Competes "violent fight" prompts against "cars on road"
        # / "normal scene" prompts so traffic footage doesn't fire.
        self._frame_clf = FrameViolenceClassifier(
            device=settings.device,
            per_frame_threshold=float(violence_cfg.get("frame_per_frame_threshold", 0.55)),
            strong_threshold=float(violence_cfg.get("frame_strong_threshold", 0.75)),
            strong_normal_max=float(violence_cfg.get("frame_strong_normal_max", 0.25)),
            window=int(violence_cfg.get("frame_window", 8)),
            min_positive=int(violence_cfg.get("frame_min_positive", 4)),
            run_every_n_frames=int(violence_cfg.get("frame_run_every_n", 6)),
        )

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
        self._stream_thread: Optional[threading.Thread] = None
        self._track_first_seen_frame: dict[int, int] = {}
        self._track_last_seen_frame: dict[int, int] = {}
        self._track_attrs: dict[int, tuple] = {}
        # Per-track set of plates already logged to PlateSighting — prevents
        # one row per frame; we only log on first sight of (track, plate).
        self._logged_sightings: set[tuple[int, str]] = set()
        # Per-track last frame we ran the second OCR engine on. Avoids
        # re-running PaddleOCR every frame for tracks that are already
        # strongly identified.
        self._track_last_ocr_frame: dict[int, int] = {}
        # Per-track last frame we wrote a PlateSighting UPDATE. The DB
        # UPDATE + WS push at 30 fps × N cars saturates the browser
        # event loop. Throttle to once per ~1 second per track.
        self._track_last_sight_update_frame: dict[int, int] = {}
        # Per-track consensus plate text — set the first time both OCR
        # engines agree on a string. Cheap signal of high reliability.
        self._track_consensus_plate: dict[int, str] = {}
        # Multi-frame character-level voting across all OCR reads of a
        # track. ~99 % per-character accuracy on 5+ frames, way beyond
        # any single-frame OCR — that's the production plate read.
        from src.ocr.plate_consensus import PlateConsensusEngine
        # Threshold lowered 3→2: short tracks (cars passing through quickly)
        # only generate ~5 reads, so 3 was too strict to ever vote.
        self._plate_consensus = PlateConsensusEngine(min_reads_for_consensus=2)
        # Per-track persisted sighting row id — when consensus changes
        # for an existing track we UPDATE that row rather than insert
        # a new one. This collapses the UI from a wall of partial reads
        # ("XA02NH7256, KA02NH7256, KA02NH7Z56, ...") into one row per
        # car that refines as more frames arrive.
        self._track_sighting_db_id: dict[int, int] = {}
        # Cache of the most recent annotated frame from inference. The stream
        # loop pushes this so detection boxes + plate text stay visible at
        # full MJPEG framerate even though inference runs slower.
        self._latest_annotated: Optional[np.ndarray] = None
        self._latest_annotated_lock = threading.Lock()

    def start(self) -> None:
        self.reader.start()
        # Stream thread: pushes every frame to MJPEG buffer at full speed
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True,
            name=f"stream-{self.spec.id}",
        )
        self._stream_thread.start()
        # Inference thread: processes frames at whatever speed GPU allows
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True,
            name=f"inference-{self.spec.id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4.0)
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        self.reader.stop()
        # Clean up live frame buffer
        with _live_frame_lock:
            _live_frames.pop(self.spec.id, None)

    # ------------------------------------------- stream loop (smooth MJPEG)

    def _stream_loop(self) -> None:
        """Push the most recently annotated frame at MJPEG rate.

        PERF: only re-encodes when the cached annotation has changed. The
        update_live_frame call does cv2.imencode which costs ~5-10 ms per
        push — at 25 Hz that's 250 ms/sec/camera. Skipping unchanged
        frames reclaims most of that for inference.
        """
        last_annotated_id = -1
        while not self._stop.is_set():
            with self._latest_annotated_lock:
                cached = self._latest_annotated
                cached_id = id(cached) if cached is not None else -1

            if cached is not None:
                # Only re-encode when the underlying frame object changed.
                if cached_id != last_annotated_id:
                    update_live_frame(self.spec.id, cached, quality=60)
                    last_annotated_id = cached_id
            else:
                # Cold-start fallback so the feed isn't black before first inference
                packet = self.reader.read()
                if packet is not None:
                    _, _, frame = packet
                    update_live_frame(self.spec.id, frame, quality=55)
            # 40 ms = 25 fps MJPEG ceiling. With unchanged-frame skip this
            # is the *upper* bound on encoding work, not a steady rate.
            time.sleep(0.04)

    # ------------------------------------------- inference loop (GPU work)

    def _inference_loop(self) -> None:
        """Grab the latest unprocessed frame and run full detection pipeline.
        
        Naturally skips frames it can't keep up with — no warnings, no
        backlog. The stream itself stays smooth via _stream_loop.
        """
        from src.common.metrics import frames_total
        while not self._stop.is_set():
            packet = self.reader.read_for_inference()
            if packet is None:
                time.sleep(0.005)
                continue
            frame_idx, ts, frame = packet
            frames_total.labels(camera_id=self.spec.id).inc()

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
        person_track_ids: list[Optional[int]] = []     # parallel to person_bboxes
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
                person_track_ids.append(td.track_id)

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

            # Multi-engine consensus pipeline:
            #   1. Take fast-alpr's text as ONE candidate (free — it was
            #      computed during detection).
            #   2. ALWAYS run PaddleOCR PP-OCRv5 on the same crop as a
            #      second candidate. Both are scored by
            #      confidence × format_validity; the winner is stored.
            #
            # Old behaviour blindly trusted fast-alpr at conf ≥ 0.30. That
            # produced ~30 % wrong-text reads on Indian / oblique plates
            # because fast-alpr's mobile-vit-v2 model is not tuned for
            # Indian fonts.
            from src.ocr.plate_ocr import PlateRead, _score as _plate_score
            candidates: list[PlateRead] = []

            # Candidate 1 — fast-alpr's pre-read (only if non-trivial text)
            if plate.text and plate.text_conf is not None and plate.text_conf >= 0.30:
                clean = "".join(c for c in plate.text.upper() if c.isalnum())
                if len(clean) >= 3:
                    candidates.append(PlateRead(
                        text=clean,
                        confidence=float(plate.text_conf),
                        engine="fast-alpr",
                    ))

            # PaddleOCR is expensive (~50-100 ms CPU per crop). Skip only
            # when this track has already produced a CONSENSUS read
            # (engines agreed) — fast-alpr's "high confidence" alone is
            # not a reliable trust signal. Re-run every 60 frames as a
            # sanity check in case the angle improved.
            had_consensus = self._track_consensus_plate.get(td.track_id) is not None
            re_ocr_due = (frame_idx - self._track_last_ocr_frame.get(td.track_id, -1000)) >= 30
            if (not had_consensus) or re_ocr_due:
                # ONE Paddle call per frame at the 25%-padded crop.
                # Multi-frame consensus across ~10 frames per track
                # gives us the same accuracy as multi-variant ensemble
                # at 1/10th the per-frame cost. The math:
                #   old: 3 padding × 5 variants × 2 engines = 30 calls
                #   new: 1 padding × 1 variant  × 1 engine  =  1 call
                # PaddleOCR PP-OCRv5 ~80ms CPU per call, so per plate
                # detection per frame: 80ms instead of 2400ms.
                crop = _crop_padded(frame, plate.xyxy, pad=0.25)
                if crop is not None and crop.size > 0:
                    paddle_read = self.ocr.read(crop)
                    if paddle_read is not None:
                        candidates.append(paddle_read)
                self._track_last_ocr_frame[td.track_id] = frame_idx

            if not candidates:
                continue

            # Consensus: any 2+ candidates that agree on the same
            # string is a strong signal. With 4 candidates (1 fast-alpr
            # + 3 padding-level Paddle reads), majority agreement is
            # near-impossible to fake — pin the confidence high.
            from collections import Counter
            text_votes = Counter(c.text for c in candidates if c.text)
            top_text, top_votes = text_votes.most_common(1)[0]
            if top_votes >= 2:
                voters = [c for c in candidates if c.text == top_text]
                best = PlateRead(
                    text=top_text,
                    confidence=min(0.99, max(v.confidence for v in voters) + 0.05 * top_votes),
                    engine=f"consensus({top_votes})",
                )
                self._track_consensus_plate[td.track_id] = best.text
            else:
                best = max(candidates, key=_plate_score)

            stored = normalize_plate(best.text) or best.text.strip()
            if stored:
                # Submit this single-frame read to the multi-frame
                # consensus engine. Then ALWAYS use the consensus when
                # it's available — that's our most accurate read for
                # this track.
                self._plate_consensus.submit(
                    td.track_id, stored, float(best.confidence), frame_idx,
                )
                consensus_text, consensus_conf = self._plate_consensus.consensus(td.track_id)
                if consensus_text and consensus_conf > 0:
                    final_text, final_conf = consensus_text, consensus_conf
                else:
                    final_text, final_conf = stored, float(best.confidence)

                self.rules.attach_plate_read(td.track_id, final_text, final_conf)
                self._log_plate_sighting(td, plate, final_text, float(final_conf), frame)

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

        # Annotate EVERY frame so the live feed shows what's being detected,
        # not just frames where a violation fired. We skip the placeholder
        # stop_line — those values are not calibrated for the test footage.
        annotated = annotate(
            frame, bundle, tracked,
            stop_line=None,
            signal_state=signal_state.value if signal_state else None,
            violations=events if events else None,
        )

        if events:
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

        # ── Step 3: Crowd analytics (uses raw frame for optical flow) ────
        # Crowd overlay is added on top of `annotated` inside this call.
        self._run_crowd_analytics(frame, frame_idx, person_bboxes, live_canvas=annotated)

        # ── Step 3: Violence detection ───────────────────────────────────
        if self._violence_enabled:
            self._run_violence_detection(frame, frame_idx, person_bboxes, person_track_ids)

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
        live_canvas: Optional[np.ndarray] = None,
    ) -> None:
        crowd = self.density_est.update(person_bboxes)
        flow = self.flow_analyzer.update(frame)
        flow_smooth = self.flow_analyzer.smoothed()
        stampede = self.stampede_det.assess(crowd, flow_smooth)

        # Dense-crowd correction: YOLO under-counts above ~30 persons/frame
        adjusted_count = self.dense_counter.get_adjusted_count(
            crowd.total_count, person_bboxes, frame
        )

        # ── Fruin Level-of-Service classification + safety alerts ────────
        # Density passes through Fruin's standard A-F bands. Surge,
        # counter-flow and stagnation alerts come out of the SafetyMonitor.
        fruin_level, fruin_severity, safety_alerts, density_trend = \
            self.safety_monitor.update(
                camera_id=self.spec.id,
                density=crowd.max_density,
                counter_flow_score=flow.counter_flow,
                stagnation_score=_stagnation_score(flow_smooth),
            )

        for alert in safety_alerts:
            log.warning("[%s] SAFETY %s — %s",
                        self.spec.id, alert.severity.upper(), alert.message)
            try:
                from src.api.server import push_violation_event
                push_violation_event({
                    "type": "safety_alert",
                    "alert_type": alert.alert_type,
                    "camera_id": self.spec.id,
                    "severity": alert.severity.value,
                    "fruin_level": alert.fruin_level.value,
                    "density": round(alert.density, 2),
                    "message": alert.message,
                    "metadata": alert.metadata,
                })
            except Exception:
                pass

        state = {
            "camera_id": self.spec.id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_count": adjusted_count,
            "yolo_raw_count": crowd.total_count,
            "dense_mode": self.dense_counter.mode if adjusted_count != crowd.total_count else "yolo",
            "max_density": crowd.max_density,
            "risk_level": crowd.risk_level,
            "fruin_level": fruin_level.value,
            "fruin_severity": fruin_severity.value,
            "fruin_color": FruinClassifier.hex_color(fruin_level),
            "fruin_description": FruinClassifier.description_for(fruin_level),
            "density_trend_per_min": round(density_trend * 60.0, 3),
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
            "active_alerts": [
                {
                    "type": a.alert_type, "severity": a.severity.value,
                    "message": a.message, "ts": a.timestamp,
                }
                for a in safety_alerts
            ],
        }
        update_crowd_state(self.spec.id, state)

        # Annotate live frame with crowd info (shown in MJPEG feed)
        # Draw crowd overlay on the detection-annotated frame (or raw if not provided)
        canvas = live_canvas if live_canvas is not None else frame
        self._annotate_crowd_on_live_frame(canvas, crowd, stampede)

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
        """Callback from async clip classifier — stores the latest VideoMAE
        verdict for this camera. Drives the violence state machine.
        """
        self._clip_confirmed = result.confirmed
        self._clip_score = result.score
        self._clip_label = result.label
        self._clip_last_update_frame = self._latest_inference_frame_idx
        log.info(
            "[%s] VideoMAE: top=%s score=%.3f confirmed=%s",
            result.camera_id, result.label, result.score, result.confirmed,
        )

    def _run_violence_detection(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_bboxes: list[tuple],
        person_track_ids: Optional[list] = None,
    ) -> None:
        # Track the latest frame so the async clip callback can timestamp itself.
        self._latest_inference_frame_idx = frame_idx

        # PERFORMANCE: skip the entire violence stack when no humans are
        # in frame. Violence requires people; running CLIP/VideoMAE/pose
        # on empty traffic burns ~80 ms/frame for nothing.
        if not person_bboxes:
            # Still need to age out the per-camera state so a quiet camera
            # auto-resolves when the last person leaves frame.
            from src.violence.pose_stage1 import Stage1Result as _S1
            empty_result = _S1(score=0.0, active=False, flags=[], person_count=0)
            self.incident_mgr.update(empty_result)
            return

        # 1) Pose Stage-1: fast skeleton heuristic. Used purely as an
        #    *acceleration* signal — it can request an early clip check,
        #    but it is NOT required for violence to fire.
        pose_result = self.pose_det.update(frame, frame_idx, person_bboxes)

        # 2) Periodic clip request OR fast trigger from pose Stage-1.
        #    Gate by frame quality — VideoMAE on motion-blurred / pitch-black
        #    frames is what produces phantom "RoadAccidents" classifications.
        if self._clip_clf.available:
            should_request = (
                pose_result.active
                or (frame_idx - self._last_clip_request_frame) >= self._clip_poll_interval
            )
            if should_request:
                q = self.quality_gate.assess(frame)
                if q.is_usable:
                    self._clip_clf.request_inference(self.spec.id, self._on_clip_result)
                    self._last_clip_request_frame = frame_idx
                else:
                    log.debug("[%s] skipping clip — %s", self.spec.id, q.reason)

        # 3) Per-frame ViT classifier — runs on every Nth frame (gated by
        #    quality so we don't run on motion-blurred frames).
        if self._frame_clf.available:
            q_for_frame = self.quality_gate.assess(frame)
            if q_for_frame.is_usable:
                fv = self._frame_clf.submit(self.spec.id, frame, frame_idx)
                if fv is not None and fv.is_violent:
                    log.info("[%s] FrameViT: violent=%.2f label=%s",
                             self.spec.id, fv.score, fv.label)
        # CLIP zero-shot per-frame classifier (image vs balanced prompts)
        clip_confirmed = self._frame_clf.is_confirmed(self.spec.id) if self._frame_clf.available else False
        clip_strong = self._frame_clf.is_strong_violent(self.spec.id) if self._frame_clf.available else False
        clip_score = self._frame_clf.latest_score(self.spec.id) if self._frame_clf.available else 0.0
        clip_label = self._frame_clf.latest_label(self.spec.id) if self._frame_clf.available else ""

        # 4) Decay: if VideoMAE hasn't said "violence" recently, treat as quiet.
        videomae_stale = (frame_idx - self._clip_last_update_frame) > self._clip_stale_after
        videomae_active = self._clip_confirmed and not videomae_stale

        # 5) THREE-SIGNAL FUSION (production rule)
        #
        #    Signal 1: pose Stage-1 (skeleton motion)
        #    Signal 2: CLIP zero-shot per-frame (image-vs-prompts)
        #    Signal 3: VideoMAE UCF-Crime (video temporal)
        #
        #    Each signal alone has known failure modes:
        #      - Pose alone: misses fights with non-canonical poses
        #      - CLIP alone: occasionally over-confident on choreographed motion
        #      - VideoMAE alone: confuses cars with Robbery/Stealing/Shooting
        #
        #    Production rule: VIOLENCE FIRES iff
        #      (a) CLIP is *strongly* violent on its own (high violent probs
        #          AND low normal probs — leaves no room for "cars on road"
        #          interpretation), OR
        #      (b) any 2 of {pose, CLIP, VideoMAE} agree.
        #
        #    This kills the cars-as-Vandalism / Robbery false positives we
        #    saw on test1.mp4 because CLIP prefers the "cars driving on a
        #    road" prompt and pose has no skeleton signal.
        pose_has_signal = bool(pose_result.flags)
        signals_agreeing = sum([pose_has_signal, clip_confirmed, videomae_active])
        violence_active = clip_strong or signals_agreeing >= 2

        # 6) Build a synthesised Stage1Result reflecting the fusion verdict.
        from src.violence.pose_stage1 import Stage1Result as _S1
        flags = list(pose_result.flags)
        if videomae_active and self._clip_label:
            flags.append(f"videomae:{self._clip_label}:{self._clip_score:.2f}")
        if clip_confirmed or clip_strong:
            tag = f"clip:{clip_label}:{clip_score:.2f}"
            if clip_strong:
                tag += ":STRONG"
            flags.append(tag)
        flags.append(f"signals_agreeing={signals_agreeing}")
        synthetic = _S1(
            score=max(
                pose_result.score,
                self._clip_score if videomae_active else 0.0,
                clip_score if (clip_confirmed or clip_strong) else 0.0,
            ),
            active=violence_active,
            flags=flags,
            person_count=pose_result.person_count,
        )

        changed = self.incident_mgr.update(synthetic)
        if changed is not None:
            self._persist_violence_incident(changed)
            log.warning(
                "[%s] VIOLENCE %s — score=%.3f flags=%s db_id=%s",
                self.spec.id, changed.status.upper(), changed.score,
                changed.flags, changed.db_id,
            )
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

        # Face capture: only while an incident is active and we have a DB id to link to.
        if (
            self.face_detector is not None
            and self.face_embedder is not None
            and self.incident_mgr.is_active()
            and self.incident_mgr.current_db_id() is not None
        ):
            self._capture_faces_for_incident(frame, frame_idx, person_bboxes,
                                             self.incident_mgr.current_db_id(),
                                             person_track_ids)

    def _capture_faces_for_incident(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_bboxes: list[tuple],
        incident_id: int,
        person_track_ids: Optional[list] = None,
    ) -> None:
        """Run face detection during an active incident and feed the
        candidates into the per-incident best-view buffer (top-K by FQA).

        Strategy:
          1. Every N frames, run YOLOv11n-face on the scene.
          2. Score each detected face using ISO-29794 quality components.
          3. Push it into the incident's BestViewBuffer; if it beats the
             current worst slot it replaces it.
          4. When the buffer slot accepts a NEW face we persist a
             FaceCapture row, embed it, and run the police-DB lookup.
          5. When a slot's face IS REPLACED (better view came along) we
             update the existing FaceCapture row in-place so the DB and
             UI always show the best K views without spamming inserts.

        This converts "first 8 captures, ignore the rest" into "best 8
        views across the entire incident".
        """
        from src.face.best_view import IncidentFaceBuffer, CandidateFace, score_face

        every_n = settings.face_capture_every_n_frames
        if frame_idx - self._last_face_scan_frame < every_n:
            return
        self._last_face_scan_frame = frame_idx

        try:
            faces = self.face_detector.extract_faces(
                frame, person_bboxes, track_ids=person_track_ids,
            )
        except TypeError:
            faces = self.face_detector.extract_faces(frame, person_bboxes)
        except Exception:
            log.debug("face detection failed", exc_info=True)
            return
        if not faces:
            return

        buf = self._face_buffers.setdefault(
            incident_id,
            IncidentFaceBuffer(
                per_track_capacity=self._face_per_track_capacity,
                total_capacity=self._face_total_capacity,
            ),
        )

        for f in faces:
            cand = CandidateFace(
                image=f.image,
                xyxy=f.xyxy,
                sharpness=f.quality,
                detection_conf=getattr(f, "detection_conf", 0.5),
                track_id=f.person_track_id,
                frame_idx=frame_idx,
            )
            cand.quality = score_face(cand.image, cand.detection_conf, cand.sharpness)
            kept, evicted, tid_key = buf.consider(cand)
            if not kept:
                continue
            # Either a fresh slot or a replaced one — persist accordingly.
            self._persist_face_capture(incident_id, cand, tid_key, evicted)

    def _persist_face_capture(
        self,
        incident_id: int,
        cand,
        tid_key: int,
        evicted,
    ) -> None:
        """Persist a FaceCapture row.

        Logic:
          - If `evicted` is None, this is a fresh slot — INSERT.
          - If `evicted` is set, the candidate replaced a worse face *of
            the same person* (or, if cross-track eviction happened to
            make room for a new participant, the worst face overall).
            We UPDATE that row in place so the Faces tab always shows
            the current best view of each participant without DB bloat.
        """
        try:
            crop_path = self.store.save_face_crop(self.spec.id, cand.image)
        except Exception:
            log.debug("face crop save failed", exc_info=True)
            return

        emb = None
        try:
            emb = self.face_embedder.embed(cand.image)
        except Exception:
            log.debug("face embedding failed", exc_info=True)

        match = None
        if emb is not None and self.police_db is not None:
            try:
                match = self.police_db.search(emb)
            except Exception:
                log.debug("police DB search failed", exc_info=True)

        slot_key = (incident_id, tid_key)
        track_db_ids = self._face_track_db_ids.setdefault(slot_key, [])

        try:
            from src.common.db import session_scope
            from src.evidence.models import FaceCapture
            with session_scope() as s:
                row = None
                is_update = False
                # If we have any stored rows for this (incident, track) and
                # we just evicted a face from this track, update its lowest-q row.
                if evicted is not None and track_db_ids:
                    rows = (
                        s.query(FaceCapture)
                         .filter(
                             FaceCapture.incident_id == incident_id,
                             FaceCapture.track_id == cand.track_id,
                         )
                         .order_by(FaceCapture.quality_score.asc())
                         .all()
                    )
                    if rows:
                        row = rows[0]
                        is_update = True
                if row is None:
                    row = FaceCapture(
                        incident_id=incident_id,
                        camera_id=self.spec.id,
                        timestamp=datetime.utcnow(),
                        track_id=cand.track_id,
                        face_crop_path=crop_path,
                        embedding=(emb.astype(np.float32).tobytes()
                                   if emb is not None else None),
                        quality_score=cand.quality,
                        matched_record_id=match.record.record_id if match else None,
                        matched_name=match.record.name if match else None,
                        matched_charges=match.record.charges if match else None,
                        matched_risk_level=match.record.risk_level if match else None,
                        match_similarity=match.similarity if match else None,
                    )
                    s.add(row)
                    s.flush()
                    track_db_ids.append(row.id)
                else:
                    row.face_crop_path = crop_path
                    row.timestamp = datetime.utcnow()
                    row.track_id = cand.track_id
                    row.embedding = (emb.astype(np.float32).tobytes()
                                     if emb is not None else None)
                    row.quality_score = cand.quality
                    row.matched_record_id = match.record.record_id if match else None
                    row.matched_name = match.record.name if match else None
                    row.matched_charges = match.record.charges if match else None
                    row.matched_risk_level = match.record.risk_level if match else None
                    row.match_similarity = match.similarity if match else None
                cand.db_id = row.id
                row_id = row.id
        except Exception:
            log.debug("face capture persist failed", exc_info=True)
            return

        try:
            from src.api.server import push_violation_event
            push_violation_event({
                "type": "face_capture",
                "id": row_id,
                "incident_id": incident_id,
                "camera_id": self.spec.id,
                "is_update": is_update,
                "quality_score": round(cand.quality, 3),
                "matched": match is not None,
                "matched_name": match.record.name if match else None,
                "matched_record_id": match.record.record_id if match else None,
                "matched_risk_level": match.record.risk_level if match else None,
                "match_similarity": round(match.similarity, 3) if match else None,
            })
        except Exception:
            pass

        if match is not None:
            log.warning(
                "[%s] FACE MATCH%s: incident=%d → %s (%s, %s) sim=%.3f q=%.2f",
                self.spec.id, " (UPDATED)" if is_update else "",
                incident_id, match.record.name,
                match.record.record_id, match.record.risk_level,
                match.similarity, cand.quality,
            )

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

            # Cache the fully-annotated frame so the stream loop pushes it.
            # We do NOT push directly here — the stream loop owns MJPEG cadence.
            with self._latest_annotated_lock:
                self._latest_annotated = frame
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
            # containment: how much of the plate is inside the vehicle box
            score = contains(vehicle.xyxy, p.xyxy)
            if score > best_score:
                best_score = score
                best = p
        # 0.20: plate centroid only needs to be within the vehicle box.
        # 0.60 was too strict — CCTV angles often have plates at bumper edge.
        return best if best_score >= 0.20 else None

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
        # Drop logged sightings for this track so memory doesn't grow forever.
        self._logged_sightings = {(t, p) for (t, p) in self._logged_sightings if t != tid}
        self._track_consensus_plate.pop(tid, None)
        self._track_last_ocr_frame.pop(tid, None)
        self._track_sighting_db_id.pop(tid, None)
        self._track_last_sight_update_frame.pop(tid, None)
        self._plate_consensus.forget(tid)
        if self.reid:
            self.reid.forget_track(self.spec.id, tid)

    # ------------------------------------------- surveillance plate logger
    _CROSS_CAM_LOOKBACK_MIN = 60   # window for "this plate was just seen elsewhere"

    def _log_plate_sighting(
        self,
        td: TrackedDetection,
        plate_det: Detection,
        plate_text: str,
        ocr_conf: float,
        frame: np.ndarray,
    ) -> None:
        """Persist a per-track plate sighting.

        One row per (camera, track_id). Subsequent reads of the same
        track UPDATE that row instead of inserting new ones, so the UI
        shows ONE plate per car that refines as more frames arrive.
        Cross-camera match flag is set when the same plate was seen on
        a different camera in the lookback window.
        """
        attrs = self._track_attrs.get(td.track_id, (None, None, None))
        global_id, color, vtype = attrs

        existing_id = self._track_sighting_db_id.get(td.track_id)
        text_key = (td.track_id, plate_text)
        text_changed = text_key not in self._logged_sightings
        if text_changed:
            self._logged_sightings.add(text_key)

        # Throttle: a track that already has a row gets at most ONE
        # UPDATE + WS push per second (~30 frames at 30 fps), unless
        # the OCR text actually changed for it. Without this throttle
        # 5 cars × 30 fps = 150 DB UPDATEs + WS broadcasts per second
        # which freezes the dashboard.
        try:
            frame_idx_now = getattr(td, "_frame_idx", None) or self._latest_inference_frame_idx
        except Exception:
            frame_idx_now = 0
        last_update_frame = self._track_last_sight_update_frame.get(td.track_id, -10000)
        if (existing_id is not None
            and not text_changed
            and (frame_idx_now - last_update_frame) < 30):
            return
        self._track_last_sight_update_frame[td.track_id] = frame_idx_now

        plate_crop_path = None
        if text_changed or existing_id is None:
            try:
                # Save the WIDER padded crop (40 % around the bbox) so the
                # operator can verify the read by eye in the UI. The tight
                # detector crop is what the OCR sees, but a clipped image
                # in the evidence column is hard for a human to confirm.
                crop = _crop_padded(frame, plate_det.xyxy, pad=0.40)
                if crop is not None and crop.size > 0:
                    plate_crop_path = self.store.save_plate_crop(plate_text, crop)
            except Exception:
                log.debug("plate crop save failed", exc_info=True)

        is_match = False
        is_new = False
        try:
            from src.common.db import session_scope
            from src.evidence.models import PlateSighting
            from sqlalchemy import select, and_

            cutoff = datetime.utcnow() - timedelta(minutes=self._CROSS_CAM_LOOKBACK_MIN)
            with session_scope() as s:
                prior = s.scalars(
                    select(PlateSighting)
                    .where(and_(
                        PlateSighting.plate_text == plate_text,
                        PlateSighting.camera_id != self.spec.id,
                        PlateSighting.timestamp >= cutoff,
                    ))
                    .order_by(PlateSighting.timestamp.desc())
                    .limit(1)
                ).first()
                is_match = prior is not None

                if existing_id is not None:
                    # UPDATE the existing per-track sighting in place
                    row = s.get(PlateSighting, existing_id)
                    if row is not None:
                        row.plate_text = plate_text
                        row.timestamp = datetime.utcnow()
                        row.ocr_confidence = ocr_conf
                        row.is_cross_camera_match = is_match
                        if plate_crop_path:
                            row.plate_crop_path = plate_crop_path
                        if global_id and not row.global_id:
                            row.global_id = global_id
                        s.flush()
                        row_id = row.id
                    else:
                        existing_id = None    # row vanished; insert a fresh one
                if existing_id is None:
                    row = PlateSighting(
                        plate_text=plate_text,
                        camera_id=self.spec.id,
                        timestamp=datetime.utcnow(),
                        track_id=td.track_id,
                        global_id=global_id,
                        ocr_confidence=ocr_conf,
                        vehicle_class=str(td.detection.cls.value) if td.detection.cls else None,
                        vehicle_color=color,
                        plate_crop_path=plate_crop_path,
                        is_cross_camera_match=is_match,
                    )
                    s.add(row)
                    s.flush()
                    row_id = row.id
                    self._track_sighting_db_id[td.track_id] = row_id
                    is_new = True
        except Exception:
            log.debug("plate sighting persist failed", exc_info=True)
            return

        # Push WS event to live surveillance feed
        try:
            from src.api.server import push_violation_event
            push_violation_event({
                "type": "plate_sighting",
                "id": row_id,
                "plate_text": plate_text,
                "camera_id": self.spec.id,
                "ocr_confidence": round(ocr_conf, 3),
                "vehicle_class": str(td.detection.cls.value) if td.detection.cls else None,
                "is_cross_camera_match": is_match,
            })
        except Exception:
            pass

        if is_match:
            log.info("[%s] CROSS-CAMERA MATCH: plate=%s seen previously on different camera",
                     self.spec.id, plate_text)

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


def _crop_padded(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    pad: float = 0.18,
) -> np.ndarray:
    """Crop with N % padding around the bbox, clipped to frame bounds.

    The plate detector emits a tight box; OCR engines need a margin of
    pixels around the glyphs to disambiguate edges from frame artefacts
    and to hold up under motion blur. 18 % is the value Indian-traffic
    OCR studies use as the sweet spot before the OCR starts confusing
    the plate frame for an additional character.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    nx1 = max(0, int(x1 - pw))
    ny1 = max(0, int(y1 - ph))
    nx2 = min(w, int(x2 + pw))
    ny2 = min(h, int(y2 + ph))
    return frame[ny1:ny2, nx1:nx2].copy()


def _stagnation_score(flow) -> float:
    """Crowd stagnation proxy from optical-flow stats.

    A dense crowd that *can't* move is the precursor to a crush. We use a
    cheap heuristic: low magnitude + low variance = the whole field is
    stuck. Returns 0..1 where >0.5 means "everyone is stationary".
    """
    if flow is None:
        return 0.0
    mag = float(getattr(flow, "magnitude", 0.0))
    var = float(getattr(flow, "variance", 0.0))
    # Normalise: magnitude < 1 px/frame and variance < 1 → fully stagnant.
    return max(0.0, min(1.0, 1.0 - (mag / 5.0) * (1.0 + var / 5.0)))


# ---------------------------------------------------------------------- top level

# Global orchestrator reference — set once during startup, used by API
_shared_orchestrator: Optional["PipelineOrchestrator"] = None


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
        # PaddleOCR PP-OCRv5 must run on CPU on Windows: paddlepaddle-gpu
        # has no usable wheel for CUDA 12.x, and passing use_gpu=True with
        # the CPU build makes paddle "switch to CPU" but leaves it in an
        # inconsistent state where predict() returns None on every call —
        # silently dropping every PaddleOCR candidate from consensus.
        # fast-alpr / onnxruntime stays on CUDA via its own provider list.
        self.ocr = PlateOCR(lang=settings.ocr_lang, use_gpu=False)
        self.store = EvidenceStore()
        self.rule_params = load_rules(settings.rules_yaml)
        # Load camera templates from yaml (but don't auto-start them)
        self.camera_templates = load_cameras(settings.cameras_yaml)

        # Expose rule_params so Settings API can mutate + persist it
        global _shared_rule_params
        with _rule_params_lock:
            _shared_rule_params = self.rule_params
        self.pipelines: dict[str, CameraPipeline] = {}
        self._camera_counter = 0
        self._lock = threading.Lock()

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

        # ── Face capture stack — only loaded when violence detection is enabled ──
        self.face_detector = None
        self.face_embedder = None
        self.police_db = None
        if self.rule_params.get("violence", {}).get("enabled", True):
            try:
                from src.face.detector import FaceDetector
                from src.face.embedder import FaceEmbedder
                from src.face.police_mock import get_police_db
                from src.face.yolo_face_detector import YoloFaceDetector

                # Prefer real YOLOv11n-face; fall back to top-30%-bbox heuristic.
                yolo_face = YoloFaceDetector(device=settings.device)
                if yolo_face.available:
                    self.face_detector = yolo_face
                    log.info("Face detector: YOLOv11n-face (WIDERFACE)")
                else:
                    self.face_detector = FaceDetector()
                    log.info("Face detector: top-30%% person-bbox heuristic (fallback)")
                # Reuse the ReID DINOv2 backbone if it's loaded — saves VRAM
                shared_backbone = self.reid.embedder if self.reid is not None else None
                self.face_embedder = FaceEmbedder(device=settings.device, shared=shared_backbone)
                self.police_db = get_police_db(
                    Path("data/police_records_mock.csv"),
                    embed_dim=self.face_embedder.DIM,
                    threshold=settings.police_match_threshold,
                )
                log.info("Face capture pipeline ready (police DB: %d records)",
                         len(self.police_db.list_all()))
            except Exception:
                log.exception("Face capture init failed — violence incidents will not capture faces")
                self.face_detector = None
                self.face_embedder = None
                self.police_db = None

        # Expose globally so API can call add/remove camera
        global _shared_orchestrator
        _shared_orchestrator = self

    def add_camera(
        self,
        source: str,
        name: str = "",
        fps_cap: int = 30,
        loop: bool = False,
        camera_id: str = "",
    ) -> str:
        """Dynamically add a camera and start its pipeline.
        
        Returns the camera_id assigned.
        """
        with self._lock:
            if not camera_id:
                self._camera_counter += 1
                camera_id = f"CAM{self._camera_counter:02d}"
                # Avoid collisions with existing pipelines
                while camera_id in self.pipelines:
                    self._camera_counter += 1
                    camera_id = f"CAM{self._camera_counter:02d}"

            if camera_id in self.pipelines:
                log.warning("Camera %s already running — ignoring add", camera_id)
                return camera_id

            # ADAPTIVE FPS: with N cameras sharing the GPU, divide the
            # frame budget so total inference load stays roughly constant.
            # 1 cam -> 30 fps, 2 cams -> 20 fps each, 3 cams -> 15 fps each,
            # 4+ -> 12 fps each (floor — below this trackers fall apart).
            n_after = len(self.pipelines) + 1
            if n_after == 1:
                effective_fps = min(fps_cap, 30)
            elif n_after == 2:
                effective_fps = min(fps_cap, 20)
            elif n_after == 3:
                effective_fps = min(fps_cap, 15)
            else:
                effective_fps = min(fps_cap, 12)

            spec = CameraSpec(
                id=camera_id,
                name=name or camera_id,
                source=source,
                fps_cap=effective_fps,
                stop_line=((100, 600), (1820, 600)),   # placeholder
                signal_roi=None,
                direction=(0.0, -1.0),
                meters_per_pixel=0.05,
                enabled=True,
                loitering_zones=[],
                crowd_zones=[],
                homography_pts=None,
            )

            log.info(
                "Adding camera %s (%s) source=%s loop=%s | adaptive fps=%d (cam #%d)",
                camera_id, name, source, loop, effective_fps, n_after,
            )

            p = CameraPipeline(
                spec, self.detector, self.ocr, self.store, self.rule_params,
                reid=self.reid,
                attr_clf=self.attr_clf,
                face_detector=self.face_detector,
                face_embedder=self.face_embedder,
                police_db=self.police_db,
            )
            # Override the stream config to support looping (with adaptive fps)
            from src.ingest.stream import StreamConfig
            p.reader = StreamReader(StreamConfig(
                camera_id=camera_id,
                source=source,
                fps_cap=effective_fps,
                loop=loop,
            ))
            p.start()
            self.pipelines[camera_id] = p
            log.info("Camera %s started successfully", camera_id)
            return camera_id

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera pipeline. Returns True if found."""
        with self._lock:
            p = self.pipelines.pop(camera_id, None)
            if p is None:
                return False
            log.info("Removing camera %s", camera_id)
            p.stop()
            return True

    def list_cameras(self) -> list[str]:
        """Return list of currently running camera IDs."""
        with self._lock:
            return list(self.pipelines.keys())

    def run(self) -> None:
        """Main loop — starts empty, cameras are added via API."""
        log.info("Pipeline orchestrator ready — no cameras started (add via UI)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Shutting down")
            with self._lock:
                for p in self.pipelines.values():
                    p.stop()

