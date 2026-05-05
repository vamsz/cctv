"""Rules engine — per-camera state machine that fires ViolationEvents.

Violations: helmet, red-light, plate-unreadable, wrong-way, triple-riding,
overspeed, watchlist-hit.

All magic numbers live in config/rules.yaml. Adding a new violation type
means writing one new _check_* method and wiring it in step().
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
from .geometry import contains, signed_distance_to_line
from .types import ViolationCode, ViolationEvent
from . import watchlist as wl


@dataclass
class _TrackState:
    last_cx: Optional[float] = None
    last_cy: Optional[float] = None
    last_signed_dist: Optional[float] = None
    no_helmet_streak: int = 0
    wrong_way_streak: int = 0
    triple_riding_streak: int = 0
    frames_seen: int = 0
    # velocity history for speed estimation (px/frame)
    vx_history: deque = field(default_factory=lambda: deque(maxlen=10))
    vy_history: deque = field(default_factory=lambda: deque(maxlen=10))
    best_plate_text: Optional[str] = None
    best_plate_conf: float = 0.0
    last_violation_at: dict[ViolationCode, float] = field(default_factory=dict)
    fps: float = 15.0


@dataclass
class CameraGeometry:
    stop_line_p1: tuple[float, float]
    stop_line_p2: tuple[float, float]
    direction: tuple[float, float]           # lawful travel unit vector
    meters_per_pixel: float = 0.05           # calibrated via calibrate.py --speed


class RulesEngine:
    def __init__(
        self,
        camera_id: str,
        geometry: CameraGeometry,
        rule_params: dict,
        fps: float = 15.0,
    ):
        self.camera_id = camera_id
        self.geo = geometry
        self.params = rule_params
        self.fps = fps
        self._state: dict[int, _TrackState] = defaultdict(lambda: _TrackState(fps=fps))
        self._signal_history: deque[SignalState] = deque(maxlen=30)

    # ------------------------------------------------------------------ entry

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
        person_dets = bundle.of(ObjectClass.PERSON)

        active_track_ids = set()
        for td in tracked:
            active_track_ids.add(td.track_id)
            st = self._state[td.track_id]
            st.frames_seen += 1

            # update velocity history
            if st.last_cx is not None:
                st.vx_history.append(td.detection.cx - st.last_cx)
                st.vy_history.append(td.detection.cy - st.last_cy)

            # --- helmet ---
            if self.params.get("helmet", {}).get("enabled") and td.detection.cls is ObjectClass.TWO_WHEELER:
                ev = self._check_helmet(td, helmet_dets, no_helmet_dets, bundle)
                if ev:
                    events.append(ev)

            # --- triple riding ---
            if self.params.get("triple_riding", {}).get("enabled") and td.detection.cls is ObjectClass.TWO_WHEELER:
                ev = self._check_triple_riding(td, person_dets, bundle)
                if ev:
                    events.append(ev)

            # --- red light ---
            if self.params.get("red_light", {}).get("enabled") and td.detection.cls in VEHICLE_CLASSES:
                ev = self._check_red_light(td, bundle)
                if ev:
                    events.append(ev)

            # --- wrong way ---
            if self.params.get("wrong_way", {}).get("enabled") and td.detection.cls in VEHICLE_CLASSES:
                ev = self._check_wrong_way(td, bundle)
                if ev:
                    events.append(ev)

            # --- overspeed ---
            if self.params.get("overspeed", {}).get("enabled") and td.detection.cls in VEHICLE_CLASSES:
                ev = self._check_overspeed(td, bundle)
                if ev:
                    events.append(ev)

            # --- watchlist ---
            if self.params.get("watchlist", {}).get("enabled"):
                ev = self._check_watchlist(td, bundle)
                if ev:
                    events.append(ev)

            # update position state
            st.last_cx, st.last_cy = td.detection.cx, td.detection.cy
            st.last_signed_dist = signed_distance_to_line(
                (td.detection.cx, td.detection.cy),
                self.geo.stop_line_p1, self.geo.stop_line_p2,
            )

        return events

    # ---------------------------------------------------------------- helmet

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

        # check upper half of bike bounding box for head detections
        upper = (veh_box[0], veh_box[1], veh_box[2], veh_box[1] + 0.5 * (veh_box[3] - veh_box[1]))
        offender_no_helmet = next(
            (d for d in no_helmet_dets if contains(upper, d.xyxy) > 0.4 and d.conf >= cfg["helmet_conf"]), None
        )
        offender_helmet = next(
            (d for d in helmet_dets if contains(upper, d.xyxy) > 0.4 and d.conf >= cfg["helmet_conf"]), None
        )

        if offender_helmet and not offender_no_helmet:
            st.no_helmet_streak = 0
            return None
        if offender_no_helmet:
            st.no_helmet_streak += 1
        else:
            return None

        if st.no_helmet_streak < cfg["min_consecutive_no_helmet_frames"]:
            return None
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

    # ---------------------------------------------------------- triple riding

    def _check_triple_riding(
        self,
        bike: TrackedDetection,
        person_dets: list[Detection],
        bundle: DetectionBundle,
    ) -> Optional[ViolationEvent]:
        st = self._state[bike.track_id]
        cfg = self.params["triple_riding"]
        bike_box = bike.detection.xyxy

        riders = [p for p in person_dets if contains(bike_box, p.xyxy) >= cfg["person_bike_overlap"]]
        if len(riders) >= cfg["min_rider_count"]:
            st.triple_riding_streak += 1
        else:
            st.triple_riding_streak = 0
            return None

        if st.triple_riding_streak < cfg["min_consecutive_frames"]:
            return None
        if self._on_cooldown(bike.track_id, ViolationCode.TRIPLE_RIDING, cfg["per_track_cooldown_seconds"]):
            return None
        self._stamp(bike.track_id, ViolationCode.TRIPLE_RIDING)
        return ViolationEvent(
            code=ViolationCode.TRIPLE_RIDING,
            camera_id=self.camera_id,
            track_id=bike.track_id,
            timestamp=bundle.timestamp,
            frame_idx=bundle.frame_idx,
            confidence=min(1.0, len(riders) / 3.0),
            plate_text=st.best_plate_text,
            plate_ocr_confidence=st.best_plate_conf or None,
            bbox=bike_box,
            extras={"rider_count": len(riders)},
        )

    # --------------------------------------------------------------- red light

    def _check_red_light(self, vehicle: TrackedDetection, bundle: DetectionBundle) -> Optional[ViolationEvent]:
        st = self._state[vehicle.track_id]
        cfg = self.params["red_light"]

        if st.last_signed_dist is None:
            return None

        cur_signed = signed_distance_to_line(
            (vehicle.detection.cx, vehicle.detection.cy),
            self.geo.stop_line_p1, self.geo.stop_line_p2,
        )
        crossed = (st.last_signed_dist <= 0 < cur_signed) or (st.last_signed_dist >= 0 > cur_signed)
        if not crossed:
            return None

        if st.last_cx is None:
            return None
        dx = vehicle.detection.cx - st.last_cx
        dy = vehicle.detection.cy - st.last_cy
        proj = dx * self.geo.direction[0] + dy * self.geo.direction[1]
        if abs(proj) < cfg["min_speed_px_per_frame"] or proj <= 0:
            return None

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

    # --------------------------------------------------------------- wrong way

    def _check_wrong_way(self, vehicle: TrackedDetection, bundle: DetectionBundle) -> Optional[ViolationEvent]:
        st = self._state[vehicle.track_id]
        cfg = self.params["wrong_way"]

        if st.last_cx is None:
            return None

        dx = vehicle.detection.cx - st.last_cx
        dy = vehicle.detection.cy - st.last_cy
        speed = (dx ** 2 + dy ** 2) ** 0.5
        if speed < cfg["min_speed_px_per_frame"]:
            return None

        # dot product with lawful direction: negative = moving against it
        proj = dx * self.geo.direction[0] + dy * self.geo.direction[1]
        if proj < -cfg["min_speed_px_per_frame"]:
            st.wrong_way_streak += 1
        else:
            st.wrong_way_streak = 0
            return None

        if st.wrong_way_streak < cfg["min_frames_wrong_way"]:
            return None
        if self._on_cooldown(vehicle.track_id, ViolationCode.WRONG_WAY, cfg["per_track_cooldown_seconds"]):
            return None
        self._stamp(vehicle.track_id, ViolationCode.WRONG_WAY)
        return ViolationEvent(
            code=ViolationCode.WRONG_WAY,
            camera_id=self.camera_id,
            track_id=vehicle.track_id,
            timestamp=bundle.timestamp,
            frame_idx=bundle.frame_idx,
            confidence=min(1.0, st.wrong_way_streak / 20.0),
            plate_text=st.best_plate_text,
            plate_ocr_confidence=st.best_plate_conf or None,
            bbox=vehicle.detection.xyxy,
        )

    # --------------------------------------------------------------- overspeed

    def _check_overspeed(self, vehicle: TrackedDetection, bundle: DetectionBundle) -> Optional[ViolationEvent]:
        st = self._state[vehicle.track_id]
        cfg = self.params["overspeed"]

        if len(st.vx_history) < cfg["min_frames_for_estimate"]:
            return None

        avg_vx = sum(st.vx_history) / len(st.vx_history)
        avg_vy = sum(st.vy_history) / len(st.vy_history)
        speed_px_per_frame = (avg_vx ** 2 + avg_vy ** 2) ** 0.5

        mpp = self.geo.meters_per_pixel or cfg.get("default_meters_per_pixel", 0.05)
        speed_ms = speed_px_per_frame * mpp * self.fps
        speed_kmh = speed_ms * 3.6

        if speed_kmh < cfg["threshold_kmh"]:
            return None
        if self._on_cooldown(vehicle.track_id, ViolationCode.OVERSPEED, cfg["per_track_cooldown_seconds"]):
            return None
        self._stamp(vehicle.track_id, ViolationCode.OVERSPEED)
        return ViolationEvent(
            code=ViolationCode.OVERSPEED,
            camera_id=self.camera_id,
            track_id=vehicle.track_id,
            timestamp=bundle.timestamp,
            frame_idx=bundle.frame_idx,
            confidence=min(1.0, (speed_kmh - cfg["threshold_kmh"]) / cfg["threshold_kmh"]),
            plate_text=st.best_plate_text,
            plate_ocr_confidence=st.best_plate_conf or None,
            bbox=vehicle.detection.xyxy,
            extras={"speed_kmh": round(speed_kmh, 1)},
        )

    # --------------------------------------------------------------- watchlist

    def _check_watchlist(self, vehicle: TrackedDetection, bundle: DetectionBundle) -> Optional[ViolationEvent]:
        st = self._state[vehicle.track_id]
        if not st.best_plate_text:
            return None
        cfg = self.params.get("watchlist", {})
        match_mode = cfg.get("match_mode", "prefix")
        hit, reason = wl.is_watchlisted(st.best_plate_text, match_mode=match_mode)
        if not hit:
            return None
        # fire at most once per track per session
        if self._on_cooldown(vehicle.track_id, ViolationCode.WATCHLIST_HIT, 300):
            return None
        self._stamp(vehicle.track_id, ViolationCode.WATCHLIST_HIT)
        return ViolationEvent(
            code=ViolationCode.WATCHLIST_HIT,
            camera_id=self.camera_id,
            track_id=vehicle.track_id,
            timestamp=bundle.timestamp,
            frame_idx=bundle.frame_idx,
            confidence=1.0,
            plate_text=st.best_plate_text,
            plate_ocr_confidence=st.best_plate_conf or None,
            bbox=vehicle.detection.xyxy,
            extras={"watchlist_reason": reason},
        )

    # ------------------------------------------ plate read aggregation

    def attach_plate_read(self, track_id: int, normalized_text: Optional[str], confidence: float) -> None:
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

    # ------------------------------------------------- cooldown helpers

    def _on_cooldown(self, track_id: int, code: ViolationCode, seconds: float) -> bool:
        last = self._state[track_id].last_violation_at.get(code)
        return last is not None and (time.time() - last) < seconds

    def _stamp(self, track_id: int, code: ViolationCode) -> None:
        self._state[track_id].last_violation_at[code] = time.time()
