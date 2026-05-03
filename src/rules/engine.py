"""Rules engine.

The engine receives, per frame:
  - a list of TrackedDetection (vehicles + persons with stable IDs)
  - the raw DetectionBundle (so we can access helmet/plate boxes)
  - the current SignalState
  - per-camera config (stop line, lawful direction, signal ROI)

It maintains a small per-track state machine and emits ViolationEvents
when a rule fires. State is keyed by track_id, so when the tracker
forgets a vehicle, its state is garbage-collected lazily.

We deliberately keep the rules independent and additive — adding a new
violation type means writing a new check method here, not editing the
detector or the OCR.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from src.detection.classes import ObjectClass, VEHICLE_CLASSES
from src.detection.detector import Detection, DetectionBundle
from src.signal.classifier import SignalState
from src.tracking.tracker import TrackedDetection
from .geometry import contains, iou, signed_distance_to_line
from .types import ViolationCode, ViolationEvent


@dataclass
class _TrackState:
    last_cx: Optional[float] = None
    last_cy: Optional[float] = None
    last_signed_dist: Optional[float] = None
    no_helmet_streak: int = 0
    frames_seen: int = 0
    best_plate_text: Optional[str] = None        # normalized
    best_plate_conf: float = 0.0
    last_violation_at: dict[ViolationCode, float] = field(default_factory=dict)


@dataclass
class CameraGeometry:
    stop_line_p1: tuple[float, float]
    stop_line_p2: tuple[float, float]
    direction: tuple[float, float]                # lawful travel unit-ish vector


class RulesEngine:
    def __init__(self, camera_id: str, geometry: CameraGeometry, rule_params: dict):
        self.camera_id = camera_id
        self.geo = geometry
        self.params = rule_params
        self._state: dict[int, _TrackState] = defaultdict(_TrackState)
        self._signal_history: deque[SignalState] = deque(maxlen=30)

    # -------------------------------------------------------------- entry point

    def step(
        self,
        tracked: list[TrackedDetection],
        bundle: DetectionBundle,
        signal_state: SignalState,
    ) -> list[ViolationEvent]:
        self._signal_history.append(signal_state)
        events: list[ViolationEvent] = []

        helmet_dets = bundle.of(ObjectClass.HELMET)
        no_helmet_dets = bundle.of(ObjectClass.NO_HELMET)
        plate_dets = bundle.of(ObjectClass.LICENSE_PLATE)

        active_track_ids = set()
        for td in tracked:
            active_track_ids.add(td.track_id)
            st = self._state[td.track_id]
            st.frames_seen += 1

            # ---- helmet rule --------------------------------------------------
            if self.params["helmet"]["enabled"] and td.detection.cls is ObjectClass.TWO_WHEELER:
                ev = self._check_helmet(td, helmet_dets, no_helmet_dets, bundle)
                if ev:
                    events.append(ev)

            # ---- red light rule ----------------------------------------------
            if self.params["red_light"]["enabled"] and td.detection.cls in VEHICLE_CLASSES:
                ev = self._check_red_light(td, bundle)
                if ev:
                    events.append(ev)

            # update last known position
            st.last_cx, st.last_cy = td.detection.cx, td.detection.cy
            st.last_signed_dist = signed_distance_to_line(
                (td.detection.cx, td.detection.cy), self.geo.stop_line_p1, self.geo.stop_line_p2
            )

        # GC stale tracks
        for tid in list(self._state.keys()):
            if tid not in active_track_ids:
                # leave for one extra frame in case track reappears next call
                pass

        return events

    # ----------------------------------------------------- individual rule logic

    def _check_helmet(
        self,
        rider: TrackedDetection,
        helmet_dets: list[Detection],
        no_helmet_dets: list[Detection],
        bundle: DetectionBundle,
    ) -> Optional[ViolationEvent]:
        st = self._state[rider.track_id]
        cfg = self.params["helmet"]
        veh_box = rider.detection.xyxy

        # Does ANY no_helmet box overlap the upper half of the rider's bike?
        upper = (veh_box[0], veh_box[1], veh_box[2], veh_box[1] + 0.5 * (veh_box[3] - veh_box[1]))
        offender_no_helmet = next(
            (d for d in no_helmet_dets if contains(upper, d.xyxy) > 0.4 and d.conf >= cfg["helmet_conf"]),
            None,
        )
        offender_helmet = next(
            (d for d in helmet_dets if contains(upper, d.xyxy) > 0.4 and d.conf >= cfg["helmet_conf"]),
            None,
        )

        if offender_helmet and not offender_no_helmet:
            st.no_helmet_streak = 0
            return None

        if offender_no_helmet:
            st.no_helmet_streak += 1
        else:
            # No info this frame — neither helmet nor no_helmet — keep streak
            return None

        if st.no_helmet_streak < cfg["min_consecutive_no_helmet_frames"]:
            return None

        # cooldown: don't fire repeatedly for the same track
        if self._on_cooldown(rider.track_id, ViolationCode.NO_HELMET, 60):
            return None
        self._stamp(rider.track_id, ViolationCode.NO_HELMET)

        return ViolationEvent(
            code=ViolationCode.NO_HELMET,
            camera_id=self.camera_id,
            track_id=rider.track_id,
            timestamp=bundle.timestamp,
            frame_idx=bundle.frame_idx,
            confidence=offender_no_helmet.conf,
            plate_text=st.best_plate_text,
            plate_ocr_confidence=st.best_plate_conf or None,
            bbox=veh_box,
        )

    def _check_red_light(self, vehicle: TrackedDetection, bundle: DetectionBundle) -> Optional[ViolationEvent]:
        st = self._state[vehicle.track_id]
        cfg = self.params["red_light"]

        # Need a "before" position to detect a crossing.
        if st.last_signed_dist is None:
            return None

        cur_signed = signed_distance_to_line(
            (vehicle.detection.cx, vehicle.detection.cy),
            self.geo.stop_line_p1,
            self.geo.stop_line_p2,
        )

        # Crossed the line: sign changed from one side to the other.
        crossed = (st.last_signed_dist <= 0 < cur_signed) or (st.last_signed_dist >= 0 > cur_signed)
        if not crossed:
            return None

        # Must be moving in the lawful travel direction (otherwise it's a
        # vehicle approaching from the opposite side; not our jurisdiction).
        if st.last_cx is None:
            return None
        dx = vehicle.detection.cx - st.last_cx
        dy = vehicle.detection.cy - st.last_cy
        # dot with negative direction == approaching from the lawful side
        # toward and beyond the stop line.
        proj = dx * self.geo.direction[0] + dy * self.geo.direction[1]
        if abs(proj) < cfg["min_speed_px_per_frame"]:
            return None
        # Lawful direction crossing: vehicle moves WITH the direction vector.
        if proj <= 0:
            return None

        # Signal must have been red for at least debounce_frames.
        recent = list(self._signal_history)[-cfg["red_debounce_frames"]:]
        if len(recent) < cfg["red_debounce_frames"] or not all(s is SignalState.RED for s in recent):
            return None

        if self._on_cooldown(vehicle.track_id, ViolationCode.RED_LIGHT_JUMP, cfg["per_track_cooldown_seconds"]):
            return None
        self._stamp(vehicle.track_id, ViolationCode.RED_LIGHT_JUMP)

        return ViolationEvent(
            code=ViolationCode.RED_LIGHT_JUMP,
            camera_id=self.camera_id,
            track_id=vehicle.track_id,
            timestamp=bundle.timestamp,
            frame_idx=bundle.frame_idx,
            confidence=vehicle.detection.conf,
            plate_text=st.best_plate_text,
            plate_ocr_confidence=st.best_plate_conf or None,
            bbox=vehicle.detection.xyxy,
        )

    # --------------------------------------------------- plate read aggregation

    def attach_plate_read(self, track_id: int, normalized_text: Optional[str], confidence: float) -> None:
        """Called by the pipeline whenever it OCRs a plate it associated to a track."""
        if not normalized_text:
            return
        st = self._state[track_id]
        if confidence > st.best_plate_conf:
            st.best_plate_text = normalized_text
            st.best_plate_conf = confidence

    def best_plate(self, track_id: int) -> tuple[Optional[str], float]:
        st = self._state.get(track_id)
        if not st:
            return None, 0.0
        return st.best_plate_text, st.best_plate_conf

    # --------------------------------------------------------- cooldown helpers

    def _on_cooldown(self, track_id: int, code: ViolationCode, seconds: float) -> bool:
        last = self._state[track_id].last_violation_at.get(code)
        return last is not None and (time.time() - last) < seconds

    def _stamp(self, track_id: int, code: ViolationCode) -> None:
        self._state[track_id].last_violation_at[code] = time.time()
