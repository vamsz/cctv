"""Frame quality assessment — gates expensive inference.

A poor-quality frame (motion blurred, very dark, washed out) is far more
likely to make a video classifier hallucinate "RoadAccidents" or other
catch-all classes. Running VideoMAE on the frame still costs ~250 ms of
GPU time but produces no useful signal.

This module gives the pipeline a fast (< 1 ms) classical-CV gate so we
only spend GPU on frames worth analysing.

Adapted from drone_surv/src/pipeline/preprocessing.py — stripped down
because fixed CCTV doesn't need the dehazer / low-light enhancer; we
just *reject* the frame and wait for a better one.

References:
  - Pertuz et al. "Analysis of focus measure operators" — Laplacian
    variance is the cheapest reliable focus/blur metric.
  - Drone_surv research.md §3 — quality gating before counting reduces
    false-positive crowd density in haze / smoke.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityScore:
    sharpness: float        # Laplacian variance — higher is sharper
    brightness: float       # mean luma 0-255
    is_usable: bool
    reason: str = ""        # populated when is_usable=False


class FrameQualityGate:
    """Cheap classical-CV gate. Tuned for fixed indoor / outdoor CCTV.

    Defaults:
      - min_sharpness = 30   (motion-blur threshold; below = severely blurred)
      - dark_below    = 20   (effectively pitch black — useless to classify)
      - bright_above  = 245  (overexposed white-out)
    """

    def __init__(
        self,
        min_sharpness: float = 30.0,
        dark_below: float = 20.0,
        bright_above: float = 245.0,
    ):
        self.min_sharpness = min_sharpness
        self.dark_below = dark_below
        self.bright_above = bright_above

    def assess(self, frame: np.ndarray) -> QualityScore:
        if frame is None or frame.size == 0:
            return QualityScore(0.0, 0.0, False, "empty frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        if sharpness < self.min_sharpness:
            return QualityScore(sharpness, brightness, False,
                                f"blurred (laplacian_var={sharpness:.0f} < {self.min_sharpness:.0f})")
        if brightness < self.dark_below:
            return QualityScore(sharpness, brightness, False,
                                f"too dark (mean={brightness:.0f})")
        if brightness > self.bright_above:
            return QualityScore(sharpness, brightness, False,
                                f"overexposed (mean={brightness:.0f})")

        return QualityScore(sharpness, brightness, True)


_singleton: FrameQualityGate | None = None


def default_gate() -> FrameQualityGate:
    global _singleton
    if _singleton is None:
        _singleton = FrameQualityGate()
    return _singleton
