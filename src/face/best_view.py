"""Best-view face selector — top-K FQA-ranked face buffer per incident.

Replaces the naive "save the sharpest face in the current frame" rule
with a multi-frame top-K buffer ranked by ISO/IEC TR 29794-5:2010
quality attributes:

  - Sharpness (Laplacian variance)
  - Resolution (face crop pixel area)
  - Detection confidence (YOLOv11n-face score)
  - Frontal-pose proxy (aspect ratio + symmetry)

When a fight runs for several seconds at 30 fps the system sees ~30
candidate faces per participant. Without best-view ranking the early
ones (often blurred / occluded by the action onset) get persisted
permanently while better later frames are discarded for being "over
the cap". Top-K replacement fixes this: any new face that scores
higher than the current worst displaces it; at incident close we
write the K best to the DB.

References:
  - ISO/IEC TR 29794-5:2010 — face image quality attributes
  - Nikitin et al., "Face Quality Assessment for Face Verification in
    Video" — sharpness + frontal weighting from skeletons
  - SFIQA-Bench (arXiv 2602.07403, 2026) — modern multi-dim FQA benchmark
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(eq=False)
class CandidateFace:
    """A single face crop + its quality score and provenance.

    eq=False is critical: the default dataclass __eq__ compares EVERY
    field, including the numpy `image` array. Comparing arrays of
    different shapes raises ValueError (broadcast). The buffer needs
    to call list.remove(face), which uses ==, so we fall back to
    identity comparison instead.
    """
    image: np.ndarray
    xyxy: tuple[float, float, float, float]
    sharpness: float
    detection_conf: float
    track_id: Optional[int]
    frame_idx: int
    quality: float = 0.0
    db_id: Optional[int] = None


_HARD_REJECT_BELOW = 0.12    # below this combined score the face is unusable


def score_face(
    crop: np.ndarray,
    detection_conf: float,
    sharpness: Optional[float] = None,
) -> float:
    """Combined ISO-29794-style face-image-quality score in [0, 1].

    Components:
      - sharpness (40 %): Laplacian variance, normalised to a sigmoid
      - size      (25 %): min(area / 200×200, 1.0) — bigger is more useful
      - detection (20 %): YOLO face confidence as-is
      - frontal   (15 %): aspect ratio penalty for tall skinny crops +
                          left/right pixel symmetry
    """
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return 0.0

    # Sharpness — Laplacian variance, sigmoid-normalised so 200+ ≈ 1.0
    if sharpness is None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_norm = float(1.0 - np.exp(-sharpness / 80.0))

    # Size — square area normalised to 200×200 reference (good crop)
    size_norm = float(min((h * w) / (200.0 * 200.0), 1.0))

    # Frontal proxy 1: aspect ratio. Faces are roughly 0.7-1.0 (W/H).
    # Penalise extremes (head turned: narrow; tilted: wide).
    ar = w / max(h, 1)
    if 0.65 <= ar <= 1.05:
        ar_score = 1.0
    elif 0.45 <= ar < 0.65 or 1.05 < ar <= 1.4:
        ar_score = 0.6
    else:
        ar_score = 0.2

    # Frontal proxy 2: left-right intensity symmetry (low diff = more frontal)
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        left = gray[:, : w // 2]
        right = cv2.flip(gray[:, w // 2:], 1)
        m = min(left.shape[1], right.shape[1])
        if m > 0:
            diff = float(np.mean(np.abs(left[:, :m].astype(np.int32) -
                                       right[:, :m].astype(np.int32))))
            sym_score = float(np.clip(1.0 - (diff / 80.0), 0.0, 1.0))
        else:
            sym_score = 0.5
    except Exception:
        sym_score = 0.5

    frontal_score = 0.5 * ar_score + 0.5 * sym_score

    return (
        0.40 * sharp_norm
        + 0.25 * size_norm
        + 0.20 * float(np.clip(detection_conf, 0.0, 1.0))
        + 0.15 * frontal_score
    )


@dataclass
class BestViewBuffer:
    """Top-K face buffer ordered by quality score."""
    capacity: int = 3
    _faces: list[CandidateFace] = field(default_factory=list)

    def consider(self, candidate: CandidateFace) -> tuple[bool, Optional[CandidateFace]]:
        """Try to add the candidate. Returns (kept, evicted).
        `evicted` is the displaced face when a slot was replaced, or None
        when the candidate filled an empty slot or was rejected.

        Hard reject anything below `_HARD_REJECT_BELOW` so non-faces
        (legs, torsos, blurry hair) never reach the DB even on cold start.
        """
        if candidate.quality <= 0.0:
            candidate.quality = score_face(
                candidate.image, candidate.detection_conf, candidate.sharpness,
            )
        if candidate.quality < _HARD_REJECT_BELOW:
            return False, None
        if len(self._faces) < self.capacity:
            self._faces.append(candidate)
            return True, None
        worst_i = min(range(len(self._faces)), key=lambda i: self._faces[i].quality)
        worst = self._faces[worst_i]
        if candidate.quality > worst.quality:
            self._faces[worst_i] = candidate
            return True, worst
        return False, None

    def best(self) -> Optional[CandidateFace]:
        if not self._faces:
            return None
        return max(self._faces, key=lambda f: f.quality)

    def all_sorted(self) -> list[CandidateFace]:
        return sorted(self._faces, key=lambda f: f.quality, reverse=True)

    def __len__(self) -> int:
        return len(self._faces)


class IncidentFaceBuffer:
    """Per-track top-K face buffer for one incident.

    Solves the problem where a single per-incident buffer was evicting
    person B's faces every time person A had a better frame. Each
    unique track_id (i.e. each detected participant) gets up to
    `per_track_capacity` slots; we cap the total faces persisted at
    `total_capacity` to bound DB writes.

    When a new candidate's track_id has its own buffer that's full,
    only the WORST face in *that track's* slots can be replaced — never
    a face of a different person.
    """

    def __init__(self, per_track_capacity: int = 3, total_capacity: int = 8):
        self.per_track_capacity = per_track_capacity
        self.total_capacity = total_capacity
        # track_id (or -1 if unknown) -> BestViewBuffer
        self._tracks: dict[int, BestViewBuffer] = {}

    def consider(self, candidate: CandidateFace) -> tuple[bool, Optional[CandidateFace], int]:
        """Returns (kept, evicted, track_key)."""
        tid_key = candidate.track_id if candidate.track_id is not None else -1
        # If we'd exceed the total cap and this track has no slots yet, only
        # accept when this candidate beats the worst existing face overall.
        if (
            tid_key not in self._tracks
            and self.total_persisted() >= self.total_capacity
        ):
            worst_track, worst_face = self._global_worst()
            if worst_face is None or candidate.quality <= worst_face.quality:
                return False, None, tid_key
            # Identity-based remove: list.remove() uses == which would trip
            # the numpy broadcast issue if eq=False ever gets undone.
            faces = self._tracks[worst_track]._faces
            for i, f in enumerate(faces):
                if f is worst_face:
                    faces.pop(i)
                    break
            kept, _ = self._get_or_make(tid_key).consider(candidate)
            return kept, worst_face, tid_key

        kept, evicted = self._get_or_make(tid_key).consider(candidate)
        return kept, evicted, tid_key

    def _get_or_make(self, tid_key: int) -> BestViewBuffer:
        b = self._tracks.get(tid_key)
        if b is None:
            b = BestViewBuffer(capacity=self.per_track_capacity)
            self._tracks[tid_key] = b
        return b

    def _global_worst(self) -> tuple[int, Optional[CandidateFace]]:
        worst_face: Optional[CandidateFace] = None
        worst_track: int = -1
        for tid, buf in self._tracks.items():
            for f in buf._faces:
                if worst_face is None or f.quality < worst_face.quality:
                    worst_face = f
                    worst_track = tid
        return worst_track, worst_face

    def total_persisted(self) -> int:
        return sum(len(b) for b in self._tracks.values())

    def all_sorted(self) -> list[CandidateFace]:
        flat: list[CandidateFace] = []
        for buf in self._tracks.values():
            flat.extend(buf._faces)
        return sorted(flat, key=lambda f: f.quality, reverse=True)

    def num_tracks(self) -> int:
        return len(self._tracks)
