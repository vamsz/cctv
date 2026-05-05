"""Thread-safe cross-camera ReID embedding store.

Keeps a rolling window of embeddings (max_age_seconds).
Search uses cosine similarity on L2-normalised vectors.

FAISS is used when installed (pip install faiss-cpu or faiss-gpu) for
fast ANN search at scale. Falls back to numpy dot-product search — which
is fine for small deployments (< 2000 entries in a 5-minute window).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Entry:
    global_id: str
    embedding: np.ndarray   # (DIM,) L2-normalised
    camera_id: str
    track_id: int
    plate_text: Optional[str]
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]
    timestamp: float        # epoch seconds


class ReIDStore:
    def __init__(
        self,
        dim: int = 576,
        max_age_seconds: float = 300.0,
        cosine_threshold: float = 0.85,
    ):
        self.dim = dim
        self.max_age = max_age_seconds
        self.threshold = cosine_threshold
        self._entries: list[_Entry] = []
        self._lock = threading.RLock()

        self._faiss = None
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            pass  # falls back to numpy

    # ---------------------------------------------------------------- public

    def search(
        self,
        embedding: np.ndarray,
        exclude_camera: Optional[str] = None,
        top_k: int = 3,
    ) -> list[tuple[str, int, float, str]]:
        """
        Returns list of (camera_id, track_id, similarity, global_id) desc by sim.
        Only entries from cameras != exclude_camera with sim >= threshold.
        """
        with self._lock:
            self._prune()
            candidates = [
                e for e in self._entries
                if exclude_camera is None or e.camera_id != exclude_camera
            ]
            if not candidates:
                return []

            mat = np.stack([e.embedding for e in candidates])  # (N, DIM)
            sims = mat @ embedding                             # (N,)

            results = []
            for i, sim in enumerate(sims):
                if float(sim) >= self.threshold:
                    e = candidates[i]
                    results.append((e.camera_id, e.track_id, float(sim), e.global_id))

            results.sort(key=lambda x: -x[2])
            return results[:top_k]

    def add_or_update(
        self,
        embedding: np.ndarray,
        camera_id: str,
        track_id: int,
        plate_text: Optional[str] = None,
        vehicle_color: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        global_id: Optional[str] = None,
    ) -> str:
        """Insert or refresh entry. Returns the assigned global_id."""
        with self._lock:
            # Update existing entry for this camera+track
            for e in self._entries:
                if e.camera_id == camera_id and e.track_id == track_id:
                    e.embedding = embedding
                    e.timestamp = time.time()
                    if plate_text:
                        e.plate_text = plate_text
                    if vehicle_color:
                        e.vehicle_color = vehicle_color
                    if vehicle_type:
                        e.vehicle_type = vehicle_type
                    if global_id and e.global_id != global_id:
                        # Merge: reassign all entries with old global_id
                        old = e.global_id
                        for other in self._entries:
                            if other.global_id == old:
                                other.global_id = global_id
                    return e.global_id

            gid = global_id or str(uuid.uuid4())
            self._entries.append(_Entry(
                global_id=gid,
                embedding=embedding,
                camera_id=camera_id,
                track_id=track_id,
                plate_text=plate_text,
                vehicle_color=vehicle_color,
                vehicle_type=vehicle_type,
                timestamp=time.time(),
            ))
            return gid

    def get_by_camera_track(
        self, camera_id: str, track_id: int
    ) -> Optional[_Entry]:
        with self._lock:
            for e in self._entries:
                if e.camera_id == camera_id and e.track_id == track_id:
                    return e
            return None

    def sightings_for_global(self, global_id: str) -> list[_Entry]:
        with self._lock:
            return [e for e in self._entries if e.global_id == global_id]

    def recent_subjects(self, limit: int = 50) -> list[dict]:
        """Returns recent unique global_ids with summary for the API."""
        with self._lock:
            self._prune()
            seen: dict[str, dict] = {}
            for e in sorted(self._entries, key=lambda x: -x.timestamp):
                if e.global_id not in seen:
                    seen[e.global_id] = {
                        "global_id": e.global_id,
                        "plate_text": e.plate_text,
                        "vehicle_color": e.vehicle_color,
                        "vehicle_type": e.vehicle_type,
                        "cameras": [],
                        "first_seen": e.timestamp,
                        "last_seen": e.timestamp,
                    }
                rec = seen[e.global_id]
                if e.camera_id not in rec["cameras"]:
                    rec["cameras"].append(e.camera_id)
                rec["last_seen"] = max(rec["last_seen"], e.timestamp)
                if len(seen) >= limit:
                    break
            return list(seen.values())

    def stats(self) -> dict:
        with self._lock:
            self._prune()
            return {
                "entries": len(self._entries),
                "unique_ids": len(set(e.global_id for e in self._entries)),
                "cameras": list(set(e.camera_id for e in self._entries)),
                "backend": "faiss" if self._faiss else "numpy",
            }

    # ---------------------------------------------------------------- private

    def _prune(self) -> None:
        cutoff = time.time() - self.max_age
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
