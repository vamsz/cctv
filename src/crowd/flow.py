"""Optical flow analysis for crowd movement.

Uses Farneback dense optical flow (OpenCV, CPU). We downsample frames
4× before computing flow — this keeps the computation under 5ms/frame
while still capturing crowd-level motion patterns.

Outputs per update:
  magnitude     mean speed across the frame (px/frame at native res)
  direction     dominant flow direction (radians, 0 = right)
  variance      speed variance — spikes indicate chaotic/panic movement
  divergence    fraction of vectors pointing outward (people fleeing a centre)
  convergence   fraction of vectors pointing inward (people crowding a point)
  counter_flow  fraction moving against dominant direction (crowd collision risk)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FlowState:
    magnitude: float        # mean flow speed px/frame (native res)
    direction: float        # dominant direction in radians
    variance: float         # flow speed std dev
    divergence: float       # 0-1
    convergence: float      # 0-1
    counter_flow: float     # 0-1
    valid: bool             # False until first pair of frames available


_FLOW_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)
_SCALE = 4      # downsample factor for optical flow computation
_COUNTER_THRESH = 0.5   # cos(angle) < -threshold → counter-flow


class FlowAnalyzer:
    def __init__(self, history_len: int = 8):
        self._prev_gray: np.ndarray | None = None
        self._history: deque[FlowState] = deque(maxlen=history_len)

    def update(self, frame: np.ndarray) -> FlowState:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        small = cv2.resize(gray, (w // _SCALE, h // _SCALE))

        if self._prev_gray is None:
            self._prev_gray = small
            state = FlowState(0, 0, 0, 0, 0, 0, False)
            self._history.append(state)
            return state

        flow = cv2.calcOpticalFlowFarneback(self._prev_gray, small, None, **_FLOW_PARAMS)
        self._prev_gray = small

        # Scale back to native resolution magnitudes
        fx = flow[..., 0] * _SCALE
        fy = flow[..., 1] * _SCALE

        magnitudes = np.sqrt(fx ** 2 + fy ** 2)
        mag_mean = float(magnitudes.mean())
        mag_var  = float(magnitudes.std())

        # Dominant direction (circular mean)
        angles = np.arctan2(fy, fx)
        dom_x = float(np.cos(angles).mean())
        dom_y = float(np.sin(angles).mean())
        dominant_dir = float(np.arctan2(dom_y, dom_x))
        dom_vec = np.array([np.cos(dominant_dir), np.sin(dominant_dir)])

        # Per-vector cos angle with dominant direction
        flat_fx = fx.ravel()
        flat_fy = fy.ravel()
        mags_flat = magnitudes.ravel()
        moving = mags_flat > 0.5   # ignore near-static pixels

        if moving.sum() < 10:
            state = FlowState(mag_mean, dominant_dir, mag_var, 0, 0, 0, True)
            self._history.append(state)
            return state

        vx = flat_fx[moving]
        vy = flat_fy[moving]
        m  = mags_flat[moving]
        ux = vx / (m + 1e-6)
        uy = vy / (m + 1e-6)
        cos_with_dom = ux * dom_vec[0] + uy * dom_vec[1]

        counter = float((cos_with_dom < -_COUNTER_THRESH).mean())

        # Divergence / convergence: compute vectors relative to frame centre
        H2, W2 = small.shape
        cy, cx = H2 / 2, W2 / 2
        ys, xs = np.mgrid[0:H2, 0:W2]
        to_centre_x = (cx - xs).ravel()[moving] * _SCALE
        to_centre_y = (cy - ys).ravel()[moving] * _SCALE
        to_c_mag = np.sqrt(to_centre_x ** 2 + to_centre_y ** 2) + 1e-6
        to_c_ux  = to_centre_x / to_c_mag
        to_c_uy  = to_centre_y / to_c_mag
        cos_to_centre = ux * to_c_ux + uy * to_c_uy
        convergence  = float((cos_to_centre >  0.6).mean())
        divergence   = float((cos_to_centre < -0.6).mean())

        state = FlowState(
            magnitude=round(mag_mean, 3),
            direction=round(dominant_dir, 3),
            variance=round(mag_var, 3),
            divergence=round(divergence, 3),
            convergence=round(convergence, 3),
            counter_flow=round(counter, 3),
            valid=True,
        )
        self._history.append(state)
        return state

    def smoothed(self) -> FlowState:
        """Temporal average over history window (reduces noise)."""
        valid = [s for s in self._history if s.valid]
        if not valid:
            return FlowState(0, 0, 0, 0, 0, 0, False)
        return FlowState(
            magnitude=float(np.mean([s.magnitude for s in valid])),
            direction=float(np.mean([s.direction for s in valid])),
            variance=float(np.mean([s.variance for s in valid])),
            divergence=float(np.mean([s.divergence for s in valid])),
            convergence=float(np.mean([s.convergence for s in valid])),
            counter_flow=float(np.mean([s.counter_flow for s in valid])),
            valid=True,
        )
