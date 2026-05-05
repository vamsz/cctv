"""Fast vehicle attribute classification — colour + type.

Colour uses an HSV histogram approach calibrated for Indian road
conditions (dust haze, variable lighting, day/night). Runs in ~0.5 ms
per crop on CPU — no GPU needed.

Type is derived directly from the already-known ObjectClass, so no
inference is needed.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from src.detection.classes import ObjectClass

# ---- colour tables --------------------------------------------------------
# Hue ranges (OpenCV hue is 0-179, half the standard 0-360)
# Each entry: (name, h_lo, h_hi) for chromatic colours
# Achromatic check comes first via saturation threshold.

_CHROMA_BINS: list[tuple[str, int, int]] = [
    ("red",    0,   10),
    ("orange", 10,  20),
    ("yellow", 20,  35),
    ("green",  35,  85),
    ("cyan",   85,  100),
    ("blue",   100, 130),
    ("purple", 130, 155),
    ("pink",   155, 165),
    ("red",    165, 180),   # red wraps at 180
]

_TYPE_LABEL: dict[ObjectClass, str] = {
    ObjectClass.TWO_WHEELER: "two_wheeler",
    ObjectClass.CAR:         "car",
    ObjectClass.BUS:         "bus",
    ObjectClass.TRUCK:       "truck",
}


class VehicleAttributeClassifier:
    def __init__(self, sat_threshold: int = 35, val_low: int = 40, val_high: int = 215):
        self.sat_thresh = sat_threshold
        self.val_low = val_low
        self.val_high = val_high

    def classify(
        self,
        crop: np.ndarray,
        obj_class: Optional[ObjectClass] = None,
    ) -> dict:
        """
        Returns {"color": str, "type": str}.
        crop is a BGR numpy array of the vehicle bounding box.
        """
        color = self._classify_color(crop)
        vtype = _TYPE_LABEL.get(obj_class, "vehicle") if obj_class else "vehicle"
        return {"color": color, "type": vtype}

    def _classify_color(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return "unknown"

        # Resize to small thumbnail — fast and noise-resistant
        thumb = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Exclude very dark pixels (shadows, wheel arches, windows)
        valid = v > self.val_low
        if not np.any(valid):
            return "black"

        median_s = float(np.median(s[valid]))
        median_v = float(np.median(v[valid]))

        # Achromatic colours first
        if median_s < self.sat_thresh:
            if median_v > self.val_high:
                return "white"
            elif median_v > 160:
                return "silver"
            elif median_v > 100:
                return "gray"
            else:
                return "black"

        # Chromatic — find dominant hue among saturated pixels
        chromatic = valid & (s > self.sat_thresh)
        if not np.any(chromatic):
            return "gray"

        h_vals = h[chromatic]
        bins = np.zeros(len(_CHROMA_BINS), dtype=int)
        for i, (_, lo, hi) in enumerate(_CHROMA_BINS):
            bins[i] = int(np.sum((h_vals >= lo) & (h_vals < hi)))

        peak_idx = int(np.argmax(bins))
        color_name = _CHROMA_BINS[peak_idx][0]

        # Disambiguate orange vs brown: orange + low value = brown
        if color_name == "orange" and median_v < 120:
            return "brown"

        return color_name
