"""Zone-based crowd density estimator.

Works from YOLO person detections already produced by the main detector —
no separate dense-crowd model required. Each detection foot-point (bottom
centre of bbox) is projected to ground-plane metres via the camera
homography and binned into configurable zones.

Density = persons / zone_area_m²

For scenarios denser than ~8 persons/m² (e.g. Kumbh-level crowds) where
individual detection fails, upgrade to CSRNet/DM-Count in Step 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraCalibration


@dataclass
class ZoneDensity:
    zone_id: str
    label: str
    person_count: int
    area_m2: float
    density: float          # people/m²
    polygon_px: np.ndarray  # shape (N,2) int32


@dataclass
class CrowdState:
    camera_id: str
    total_count: int
    zones: list[ZoneDensity]
    max_density: float
    risk_score: float       # 0-1
    risk_level: str         # ok / warning / critical
    risk_flags: list[str]

    @property
    def is_dangerous(self) -> bool:
        return self.risk_level == "critical"


class ZoneDensityEstimator:
    """Per-camera crowd density tracker."""

    def __init__(
        self,
        camera_id: str,
        zones_cfg: list[dict],
        calibration: CameraCalibration,
        danger_density: float = 5.0,    # people/m² → critical
        warning_density: float = 3.0,   # people/m² → warning
        danger_count: int = 15,         # absolute count (no homography needed)
        warning_count: int = 8,
    ):
        self.camera_id = camera_id
        self.calibration = calibration
        self.danger_density = danger_density
        self.warning_density = warning_density
        self.danger_count = danger_count
        self.warning_count = warning_count

        self._zones: list[dict] = []
        for z in zones_cfg:
            poly_px = np.array(z["polygon"], dtype=np.int32)
            # Estimate zone area in m² via sampling
            area_m2 = self._estimate_zone_area_m2(poly_px)
            self._zones.append({
                "id": z["id"],
                "label": z.get("label", z["id"]),
                "polygon": poly_px,
                "area_m2": max(area_m2, 1.0),
            })

        # Default zone: whole frame
        if not self._zones:
            h, w = calibration.frame_h, calibration.frame_w
            poly = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
            area_m2 = self._estimate_zone_area_m2(poly)
            self._zones.append({
                "id": "default",
                "label": "Full Frame",
                "polygon": poly,
                "area_m2": max(area_m2, 1.0),
            })

    # ------------------------------------------------------------------

    def update(self, person_bboxes: list[tuple]) -> CrowdState:
        """
        person_bboxes: list of (x1,y1,x2,y2) tuples for detected persons.
        Returns CrowdState.
        """
        foot_points = [(int((x1 + x2) / 2), int(y2)) for x1, y1, x2, y2 in person_bboxes]

        zone_results: list[ZoneDensity] = []
        for z in self._zones:
            count = sum(
                1 for fx, fy in foot_points
                if cv2.pointPolygonTest(z["polygon"].reshape(-1, 1, 2), (float(fx), float(fy)), False) >= 0
            )
            density = count / z["area_m2"]
            zone_results.append(ZoneDensity(
                zone_id=z["id"],
                label=z["label"],
                person_count=count,
                area_m2=z["area_m2"],
                density=density,
                polygon_px=z["polygon"],
            ))

        total = len(person_bboxes)
        max_density = max((zr.density for zr in zone_results), default=0.0)

        risk_score, risk_level, risk_flags = self._assess_risk(total, max_density)

        return CrowdState(
            camera_id=self.camera_id,
            total_count=total,
            zones=zone_results,
            max_density=round(max_density, 2),
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            risk_flags=risk_flags,
        )

    # ------------------------------------------------------------------

    def _assess_risk(self, count: int, max_density: float) -> tuple[float, str, list[str]]:
        flags: list[str] = []
        score = 0.0

        if max_density >= self.danger_density:
            flags.append(f"density_critical ({max_density:.1f} p/m²)")
            score = max(score, 0.9)
        elif max_density >= self.warning_density:
            flags.append(f"density_warning ({max_density:.1f} p/m²)")
            score = max(score, 0.5)

        if count >= self.danger_count:
            flags.append(f"count_critical ({count})")
            score = max(score, 0.8)
        elif count >= self.warning_count:
            flags.append(f"count_warning ({count})")
            score = max(score, 0.4)

        if score >= 0.8:
            level = "critical"
        elif score >= 0.4:
            level = "warning"
        else:
            level = "ok"

        return score, level, flags

    def _estimate_zone_area_m2(self, poly_px: np.ndarray) -> float:
        """Sample grid points inside polygon and project to ground plane."""
        x_vals = poly_px[:, 0]
        y_vals = poly_px[:, 1]
        xs = np.linspace(x_vals.min(), x_vals.max(), 10)
        ys = np.linspace(y_vals.min(), y_vals.max(), 10)
        pts_world = []
        for x in xs:
            for y in ys:
                if cv2.pointPolygonTest(poly_px.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
                    wx, wy = self.calibration.image_to_world(x, y)
                    pts_world.append((wx, wy))
        if len(pts_world) < 3:
            return float(cv2.contourArea(poly_px.reshape(-1, 1, 2))) / 1e4
        arr = np.array(pts_world, dtype=np.float32)
        hull = cv2.convexHull(arr)
        return float(cv2.contourArea(hull))
