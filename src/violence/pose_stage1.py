"""Stage-1 violence detection via YOLO11n-pose skeleton heuristics.

Four signals, ≥ 2 active simultaneously → stage1 positive (per plan.md rule).

Signals:
  arm_raise   wrist/elbow above shoulder on multiple persons (aggressive posture)
  proximity   person bounding-box IoU ≥ threshold (physical contact)
  rapid_limb  wrist keypoint velocity ≥ threshold px/frame (punching/flailing)
  fall        nose Y ≥ hip Y midpoint (person knocked down / collapsed)

Runs only on frames where ≥ 2 persons are present, every `run_every_n` frames,
to keep GPU load minimal. Returns the last computed result on skipped frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# COCO 17-keypoint indices
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

_WEIGHTS = {"arm_raise": 0.30, "proximity": 0.25, "rapid_limb": 0.30, "fall": 0.15}


@dataclass
class Stage1Result:
    score: float
    active: bool           # True iff ≥ 2 signals AND score ≥ min_score
    flags: list[str] = field(default_factory=list)
    person_count: int = 0


class PoseViolenceDetector:
    """Per-camera Stage-1 violence detector."""

    def __init__(
        self,
        run_every_n: int = 3,
        kp_conf: float = 0.40,
        proximity_iou: float = 0.12,
        wrist_vel_thresh: float = 8.0,   # px/frame at native resolution
        min_score: float = 0.12,
        device: str = "cpu",
        weights: str = "yolo11n-pose.pt",
    ):
        self.run_every_n = run_every_n
        self.kp_conf = kp_conf
        self.proximity_iou = proximity_iou
        self.wrist_vel_thresh = wrist_vel_thresh
        self.min_score = min_score
        self._device = device
        self._weights = weights
        self._model = None
        self._frame_count = 0
        # (person_idx, kp_idx) → deque[(frame_idx, xy)]
        self._wrist_hist: dict[tuple, deque] = {}
        self._last: Stage1Result = Stage1Result(score=0.0, active=False)

    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        person_bboxes: list[tuple],
    ) -> Stage1Result:
        """
        person_bboxes: list of (x1,y1,x2,y2) for detected persons this frame.
        """
        self._frame_count += 1

        if len(person_bboxes) < 2:
            self._last = Stage1Result(score=0.0, active=False, person_count=len(person_bboxes))
            return self._last

        if self._frame_count % self.run_every_n != 0:
            return self._last

        self._load_model()
        try:
            results = self._model(frame, verbose=False, device=self._device)
        except Exception:
            return self._last

        if not results or results[0].keypoints is None:
            return self._last

        kpts_data = results[0].keypoints.data  # tensor (N, 17, 3)
        kpts = kpts_data.cpu().numpy()
        if kpts.shape[0] < 2:
            return self._last

        boxes_raw = results[0].boxes
        boxes = boxes_raw.xyxy.cpu().numpy() if boxes_raw is not None else np.zeros((0, 4))

        signals: dict[str, float] = {}
        flags: list[str] = []

        # ── arm_raise: elbow or wrist above shoulder ─────────────────
        raise_count = 0
        for kp in kpts:
            for side in ((_L_SHOULDER, _L_ELBOW, _L_WRIST), (_R_SHOULDER, _R_ELBOW, _R_WRIST)):
                sh, el, wr = kp[side[0]], kp[side[1]], kp[side[2]]
                if sh[2] < self.kp_conf or el[2] < self.kp_conf:
                    continue
                if el[1] < sh[1]:
                    raise_count += 1
                elif wr[2] >= self.kp_conf and wr[1] < sh[1]:
                    raise_count += 1
        if raise_count >= 2:
            signals["arm_raise"] = min(raise_count / max(len(kpts) * 2, 2), 1.0)
            flags.append(f"arm_raise x{raise_count}")

        # ── proximity: pairwise box IoU ──────────────────────────────
        if boxes.shape[0] >= 2:
            max_iou = 0.0
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = _box_iou(boxes[i], boxes[j])
                    if iou > max_iou:
                        max_iou = iou
            if max_iou >= self.proximity_iou:
                signals["proximity"] = min(max_iou / 0.5, 1.0)
                flags.append(f"proximity iou={max_iou:.2f}")

        # ── rapid_limb: wrist keypoint velocity ──────────────────────
        max_vel = 0.0
        for pi, kp in enumerate(kpts):
            for ki in (_L_WRIST, _R_WRIST):
                pt = kp[ki]
                if pt[2] < self.kp_conf:
                    continue
                key = (pi, ki)
                if key not in self._wrist_hist:
                    self._wrist_hist[key] = deque(maxlen=4)
                self._wrist_hist[key].append((frame_idx, pt[:2].copy()))
                hist = self._wrist_hist[key]
                if len(hist) >= 2:
                    fi0, xy0 = hist[0]
                    fi1, xy1 = hist[-1]
                    dt = max(fi1 - fi0, 1)
                    vel = float(np.linalg.norm(xy1 - xy0)) / dt
                    if vel > max_vel:
                        max_vel = vel
        if max_vel >= self.wrist_vel_thresh:
            signals["rapid_limb"] = min(max_vel / 30.0, 1.0)
            flags.append(f"rapid_limb {max_vel:.1f}px/f")

        # ── fall: nose at or below hip centre ────────────────────────
        fall_count = 0
        for kp in kpts:
            nose = kp[_NOSE]
            l_hip, r_hip = kp[_L_HIP], kp[_R_HIP]
            if nose[2] < self.kp_conf:
                continue
            hip_ys = [h[1] for h in (l_hip, r_hip) if h[2] >= self.kp_conf]
            if hip_ys and nose[1] >= (sum(hip_ys) / len(hip_ys)) - 10:
                fall_count += 1
        if fall_count >= 1:
            signals["fall"] = min(fall_count / max(len(kpts), 1), 1.0)
            flags.append(f"fall x{fall_count}")

        # ── composite ────────────────────────────────────────────────
        raw = sum(signals[k] * _WEIGHTS[k] for k in signals)
        n_signals = len(signals)
        active = n_signals >= 2 and raw >= self.min_score

        self._last = Stage1Result(
            score=round(min(raw, 1.0), 3),
            active=active,
            flags=flags,
            person_count=int(kpts.shape[0]),
        )
        return self._last

    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._weights)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)
