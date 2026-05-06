"""Cross-camera vehicle Re-Identification.

For each vehicle track, we:
  1. Extract an embedding (every extract_every_n frames to limit GPU load).
  2. Search the shared store for a match from a DIFFERENT camera.
  3. Assign the matched global_id (or create a new one).
  4. If a plate is known from the matched entry and we don't have one,
     propagate it back to the rules engine.

The CrossCameraReID instance is shared across all CameraPipelines.
Thread-safe: the store uses its own RLock.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from .embedder import VehicleEmbedder
from .store import ReIDStore


class CrossCameraReID:
    def __init__(
        self,
        device: str = "cpu",
        cosine_threshold: float = 0.85,
        max_age_seconds: float = 300.0,
        extract_every_n_frames: int = 15,
    ):
        self.embedder = VehicleEmbedder(device=device)
        self.store = ReIDStore(
            dim=self.embedder.DIM,
            max_age_seconds=max_age_seconds,
            cosine_threshold=cosine_threshold,
        )
        self.extract_every_n = extract_every_n_frames
        self._last_frame: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def process(
        self,
        frame: np.ndarray,
        xyxy: tuple[float, float, float, float],
        camera_id: str,
        track_id: int,
        frame_idx: int,
        plate_text: Optional[str] = None,
        vehicle_color: Optional[str] = None,
        vehicle_type: Optional[str] = None,
    ) -> tuple[Optional[str], float, Optional[str]]:
        """
        Returns (global_id, match_sim, propagated_plate).
        - global_id: stable cross-camera identity string (UUID)
        - match_sim: cosine similarity to best cross-camera match (0.0 = new)
        - propagated_plate: plate text from matched entry if we don't have one
        """
        key = (camera_id, track_id)

        # Check if we have an existing entry without extracting again
        with self._lock:
            last = self._last_frame.get(key, -self.extract_every_n - 1)
            due = (frame_idx - last) >= self.extract_every_n

        if not due:
            existing = self.store.get_by_camera_track(camera_id, track_id)
            if existing:
                return existing.global_id, 0.0, None
            return None, 0.0, None

        # Time to extract
        with self._lock:
            self._last_frame[key] = frame_idx

        embedding = self.embedder.extract(frame, xyxy)
        if embedding is None:
            return None, 0.0, None

        # Search for cross-camera matches
        matches = self.store.search(embedding, exclude_camera=camera_id, top_k=1)

        propagated_plate: Optional[str] = None
        if matches:
            cam_id, t_id, sim, matched_gid = matches[0]
            # If matched entry has a plate and we don't, propagate it
            if not plate_text:
                matched_entry = self.store.get_by_camera_track(cam_id, t_id)
                if matched_entry and matched_entry.plate_text:
                    propagated_plate = matched_entry.plate_text
            global_id = self.store.add_or_update(
                embedding, camera_id, track_id, plate_text,
                vehicle_color, vehicle_type, global_id=matched_gid,
            )
            return global_id, sim, propagated_plate
        else:
            global_id = self.store.add_or_update(
                embedding, camera_id, track_id, plate_text,
                vehicle_color, vehicle_type,
            )
            return global_id, 0.0, None

    def forget_track(self, camera_id: str, track_id: int) -> None:
        with self._lock:
            self._last_frame.pop((camera_id, track_id), None)

    def stats(self) -> dict:
        return self.store.stats()
