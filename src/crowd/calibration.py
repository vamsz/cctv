"""Per-camera homography calibration.

Maps image pixel coordinates → ground-plane metres so density, speed,
and distance are expressed in real-world units rather than pixels.

Calibration is loaded from cameras.yaml (`homography_pts`) or can be
set interactively via `scripts/calibrate_camera.py`.

Without calibration the estimator falls back to pixel-area approximation:
  density ≈ count / (frame_area_pixels / 1e5)

Schema in cameras.yaml:
  homography_pts:          # 4+ corresponding point pairs
    image: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]   # pixels
    world: [[X0,Y0], [X1,Y1], [X2,Y2], [X3,Y3]]   # metres from camera base
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


class CameraCalibration:
    def __init__(self, H: Optional[np.ndarray] = None, frame_h: int = 720, frame_w: int = 1280):
        self.H = H                  # 3×3 homography: image → ground (metres)
        self.H_inv = np.linalg.inv(H) if H is not None else None
        self.frame_h = frame_h
        self.frame_w = frame_w

    # ------------------------------------------------------------------
    def image_to_world(self, x: float, y: float) -> tuple[float, float]:
        """Project image pixel to ground-plane metre coords."""
        if self.H is None:
            # Fallback: normalise pixel to approximate metre equivalent
            return x / self.frame_w * 20.0, y / self.frame_h * 15.0
        p = self.H @ np.array([x, y, 1.0], dtype=np.float64)
        return float(p[0] / p[2]), float(p[1] / p[2])

    def approx_scale_m_per_px(self, y_mid: float) -> float:
        """Rough metres/pixel at image row y_mid (perspective approximation)."""
        if self.H is None:
            return 0.04
        _, Y0 = self.image_to_world(self.frame_w / 2, y_mid)
        _, Y1 = self.image_to_world(self.frame_w / 2, y_mid + 10)
        return abs(Y1 - Y0) / 10.0

    # ------------------------------------------------------------------
    @classmethod
    def from_points(
        cls,
        image_pts: list[list[float]],
        world_pts: list[list[float]],
        frame_h: int = 720,
        frame_w: int = 1280,
    ) -> "CameraCalibration":
        img = np.array(image_pts, dtype=np.float32)
        wld = np.array(world_pts, dtype=np.float32)
        H, _ = cv2.findHomography(img, wld, cv2.RANSAC)
        return cls(H=H, frame_h=frame_h, frame_w=frame_w)

    @classmethod
    def identity(cls, frame_h: int = 720, frame_w: int = 1280) -> "CameraCalibration":
        return cls(H=None, frame_h=frame_h, frame_w=frame_w)


def load_calibrations(cameras_yaml: Path) -> dict[str, CameraCalibration]:
    """Load per-camera calibrations from cameras.yaml."""
    with open(cameras_yaml) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, CameraCalibration] = {}
    for cam in data.get("cameras", []):
        cid = cam["id"]
        hpts = cam.get("homography_pts")
        if hpts and "image" in hpts and "world" in hpts:
            out[cid] = CameraCalibration.from_points(hpts["image"], hpts["world"])
        else:
            out[cid] = CameraCalibration.identity()
    return out
