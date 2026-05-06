"""Trajectory feature extraction for improved loitering detection.

Per improvement plan (Area 8) and Núñez Cano et al. 2024:
  "Identifying Loitering Behavior with Trajectory Analysis"

Two features that substantially reduce false positives:

  path_displacement_ratio
    = total_path_length / displacement_from_start
    Bus-stop standing: ratio ≈ 1 (barely moving, low path AND low displacement)
    Vendor at stall:   ratio ≈ 1-2
    Genuine loiterer:  ratio > 3 (walking back and forth without getting anywhere)
    → Flag when ratio > PATH_DISP_THRESHOLD (default 2.5)

  turn_angle_variance
    = variance of angle changes between consecutive trajectory segments
    Pacing (loitering signal): high variance (changes direction repeatedly)
    Waiting at bus stop:       low variance (standing or facing one direction)
    → Flag when variance > TURN_VAR_THRESHOLD (default 0.8 radians²)

Usage:
    store = TrajectoryStore()
    store.add_point(track_id, (cx, cy))          # call every frame
    features = store.compute_features(track_id)   # check periodically
    is_loitering = features.is_suspicious()
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Tuning constants (India: vendors + auto drivers often stand long, low ratio)
PATH_DISP_THRESHOLD = 2.5    # path/displacement > this → wandering
TURN_VAR_THRESHOLD = 0.80    # turn-angle variance > this → pacing
MIN_POINTS = 15              # need this many points before features are meaningful


@dataclass
class TrajectoryFeatures:
    track_id: int
    path_length: float          # total pixel distance travelled
    displacement: float         # straight-line distance from first to last point
    path_displacement_ratio: float
    turn_angle_variance: float  # variance of heading changes (radians²)
    n_points: int

    def is_suspicious(
        self,
        path_disp_thresh: float = PATH_DISP_THRESHOLD,
        turn_var_thresh: float = TURN_VAR_THRESHOLD,
    ) -> bool:
        """True if trajectory shows wandering OR pacing (loitering signals)."""
        if self.n_points < MIN_POINTS:
            return False
        wandering = self.path_displacement_ratio > path_disp_thresh
        pacing = self.turn_angle_variance > turn_var_thresh
        return wandering or pacing

    def exclude_bus_stop(self) -> bool:
        """True when this looks like a bus-stop wait (low path, low variance)."""
        if self.n_points < MIN_POINTS:
            return False
        return (
            self.path_displacement_ratio < 1.5
            and self.turn_angle_variance < 0.30
            and self.path_length < 50
        )


class TrajectoryStore:
    """Per-camera position history for all active tracks."""

    def __init__(self, max_len: int = 150):
        self._max_len = max_len
        # track_id → deque of (cx, cy) float tuples
        self._positions: dict[int, deque] = {}

    def add_point(self, track_id: int, point: tuple[float, float]) -> None:
        if track_id not in self._positions:
            self._positions[track_id] = deque(maxlen=self._max_len)
        self._positions[track_id].append(point)

    def forget(self, track_id: int) -> None:
        self._positions.pop(track_id, None)

    def compute_features(self, track_id: int) -> Optional[TrajectoryFeatures]:
        pts = self._positions.get(track_id)
        if pts is None or len(pts) < 4:
            return None

        arr = np.array(pts, dtype=np.float32)  # (N, 2)
        n = len(arr)

        # ── path length
        diffs = np.diff(arr, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path_length = float(np.sum(segment_lengths))

        # ── displacement (first → last)
        displacement = float(np.linalg.norm(arr[-1] - arr[0]))

        # ── path/displacement ratio (handle zero displacement)
        if displacement < 1.0:
            # Person hasn't moved at all — low ratio, not wandering
            ratio = 1.0
        else:
            ratio = path_length / displacement

        # ── turn-angle variance
        turn_var = _compute_turn_angle_variance(arr)

        return TrajectoryFeatures(
            track_id=track_id,
            path_length=round(path_length, 1),
            displacement=round(displacement, 1),
            path_displacement_ratio=round(ratio, 2),
            turn_angle_variance=round(turn_var, 4),
            n_points=n,
        )

    def active_track_ids(self) -> list[int]:
        return list(self._positions.keys())

    def cleanup(self, active_ids: set[int]) -> None:
        stale = [tid for tid in self._positions if tid not in active_ids]
        for tid in stale:
            del self._positions[tid]


def _compute_turn_angle_variance(arr: np.ndarray) -> float:
    """Variance of heading changes between consecutive trajectory segments."""
    if len(arr) < 3:
        return 0.0

    diffs = np.diff(arr, axis=0)             # (N-1, 2) segment vectors
    lengths = np.linalg.norm(diffs, axis=1)

    # Filter out near-zero segments (person stationary)
    valid = lengths > 0.5
    if np.sum(valid) < 2:
        return 0.0

    valid_diffs = diffs[valid]

    # Heading angle for each valid segment
    angles = np.arctan2(valid_diffs[:, 1], valid_diffs[:, 0])

    # Turn angles between consecutive segments
    turn_angles = np.diff(angles)

    # Wrap to [-π, π]
    turn_angles = (turn_angles + np.pi) % (2 * np.pi) - np.pi

    return float(np.var(turn_angles))
