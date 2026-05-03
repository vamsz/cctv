"""Traffic signal state classifier.

We don't need a CNN for this. The signal ROI is a known small region
(configured per camera). We threshold in HSV color space for red,
amber, and green, count the bright pixels in each band, and pick the
strongest one — falling back to UNKNOWN when no band has a clear lead.

A small CNN is more robust against sun glare and back-lit signals; we
keep this interface so it can be swapped in later without changing
callers. The rules engine just consumes a SignalState.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import cv2
import numpy as np


class SignalState(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"
    UNKNOWN = "unknown"


# HSV ranges. Red wraps around the H axis so it needs two ranges.
RED_LO_1 = np.array([0, 120, 120]);   RED_HI_1 = np.array([10, 255, 255])
RED_LO_2 = np.array([170, 120, 120]); RED_HI_2 = np.array([180, 255, 255])
AMBER_LO = np.array([15, 120, 120]);  AMBER_HI = np.array([35, 255, 255])
GREEN_LO = np.array([40, 80, 80]);    GREEN_HI = np.array([90, 255, 255])

# Minimum fraction of the ROI that must light up before we trust a color.
MIN_PIXEL_FRACTION = 0.02


class SignalClassifier:
    def __init__(self, signal_roi: Optional[tuple[int, int, int, int]] = None):
        self.roi = signal_roi  # (x1, y1, x2, y2) or None

    def classify(self, frame: np.ndarray) -> SignalState:
        if self.roi is None:
            return SignalState.UNKNOWN
        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return SignalState.UNKNOWN
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = crop.shape[0] * crop.shape[1]

        red = (cv2.inRange(hsv, RED_LO_1, RED_HI_1) | cv2.inRange(hsv, RED_LO_2, RED_HI_2)).sum() / 255
        amber = cv2.inRange(hsv, AMBER_LO, AMBER_HI).sum() / 255
        green = cv2.inRange(hsv, GREEN_LO, GREEN_HI).sum() / 255

        scores = {SignalState.RED: red, SignalState.AMBER: amber, SignalState.GREEN: green}
        winner, score = max(scores.items(), key=lambda kv: kv[1])
        if score / total < MIN_PIXEL_FRACTION:
            return SignalState.UNKNOWN
        return winner
