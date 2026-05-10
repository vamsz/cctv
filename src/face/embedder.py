"""Face crop → 384-d embedding for police-DB cosine matching.

Reuses the same DINOv2-Small backbone as the vehicle ReID embedder. We
deliberately do NOT load FaceNet here — DINOv2 is already cached in
memory after pipeline startup, and using two networks just to embed
faces would double VRAM use without a meaningful accuracy gain at the
mock-DB cosine threshold we work at.

Production note: when you upgrade the police DB to a real provider,
swap this for an ArcFace / FaceNet embedder of the same dimensionality
(or update police_mock.DIM accordingly). The matcher API stays
identical.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.reid.embedder import VehicleEmbedder

log = logging.getLogger("face.embedder")


class FaceEmbedder:
    """Thin wrapper over the vehicle DINOv2 embedder for face crops."""

    DIM = 384

    def __init__(self, device: str = "cpu", shared: Optional[VehicleEmbedder] = None):
        if shared is not None:
            self._impl = shared
            log.info("FaceEmbedder: sharing %s backbone with ReID", shared.backend)
        else:
            self._impl = VehicleEmbedder(device=device)
            log.info("FaceEmbedder: standalone %s backbone on %s", self._impl.backend, device)
        self.DIM = self._impl.DIM

    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return an L2-normalised float32 embedding for the face crop,
        or None if the crop is unusable.
        """
        return self._impl.extract_crop(crop_bgr)
