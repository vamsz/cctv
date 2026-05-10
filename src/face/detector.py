"""Face / head-region detector for surveillance face capture.

Strategy:
  1. Take each PERSON detection bounding box.
  2. Carve out the upper portion of the box as the head region. We use
     30 % by default — enough to include hair + ears for most camera
     angles while excluding shoulders / chest where face matching gets
     polluted by clothing.
  3. Reject crops that are too small or too washed-out (low Laplacian
     variance) to be useful for embedding match.

This avoids a separate face detection model — important because we
already load YOLO11 (general), helmet, fast-alpr, VideoMAE, DINOv2
and YOLO11n-pose. Adding a fifth network would push past the GPU
budget on the laptop class hardware this ships against. Pose
keypoints (when available) sharpen the head box; without them,
the upper-30 % heuristic is the fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FaceCrop:
    xyxy: tuple[float, float, float, float]
    image: np.ndarray
    quality: float                       # Laplacian variance — higher is sharper
    person_track_id: Optional[int] = None


class FaceDetector:
    """Lightweight head-region extractor.

    Parameters
    ----------
    head_fraction : float
        Fraction of person bbox height that counts as head. 0.30 covers
        head + ears for typical CCTV angles.
    min_side_px : int
        Reject crops smaller than this on either dimension — too small
        to embed reliably.
    min_quality : float
        Minimum Laplacian variance to accept the crop. Below this the
        face is too blurry to be useful.
    """

    def __init__(
        self,
        head_fraction: float = 0.30,
        min_side_px: int = 48,
        min_quality: float = 25.0,
    ):
        self.head_fraction = head_fraction
        self.min_side_px = min_side_px
        self.min_quality = min_quality

    def extract_faces(
        self,
        frame: np.ndarray,
        person_bboxes: list[tuple[float, float, float, float]],
        track_ids: Optional[list[Optional[int]]] = None,
    ) -> list[FaceCrop]:
        """Return one FaceCrop per person whose head region passes the
        size + quality gates.
        """
        if not person_bboxes:
            return []
        if track_ids is None:
            track_ids = [None] * len(person_bboxes)

        out: list[FaceCrop] = []
        h_frame, w_frame = frame.shape[:2]

        for bbox, tid in zip(person_bboxes, track_ids):
            x1, y1, x2, y2 = bbox
            ph = y2 - y1
            if ph <= 0:
                continue
            head_h = ph * self.head_fraction
            hx1 = max(0, int(x1))
            hy1 = max(0, int(y1))
            hx2 = min(w_frame, int(x2))
            hy2 = min(h_frame, int(y1 + head_h))

            if hx2 - hx1 < self.min_side_px or hy2 - hy1 < self.min_side_px:
                continue

            crop = frame[hy1:hy2, hx1:hx2]
            if crop.size == 0:
                continue

            quality = _laplacian_variance(crop)
            if quality < self.min_quality:
                continue

            out.append(FaceCrop(
                xyxy=(float(hx1), float(hy1), float(hx2), float(hy2)),
                image=crop.copy(),
                quality=float(quality),
                person_track_id=tid,
            ))

        return out


def _laplacian_variance(img: np.ndarray) -> float:
    """Cheap sharpness proxy — variance of the Laplacian. Used for
    rejecting motion-blurred / out-of-focus crops before we waste an
    embedding forward pass on them.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
