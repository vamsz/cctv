"""High-accuracy license plate OCR pipeline (EasyOCR backend).

Strategy for maximum plate read accuracy:
  1. Multiple preprocessing pipelines (grayscale, adaptive threshold,
     CLAHE contrast, morphological cleaning, etc.)
  2. Each pipeline produces a candidate; we score them and pick the best.
  3. Aggressive upscaling — OCR needs at least 30+ pixel text height.
  4. Deskew detection for slightly rotated plates.
  5. Border removal to eliminate plate frame artifacts.

We use EasyOCR rather than PaddleOCR because PaddleOCR 3.x has a
Windows-specific oneDNN compatibility bug. EasyOCR is a pure-PyTorch
implementation that works reliably across platforms.

This module is the #1 accuracy bottleneck in the system — if the plate
text is wrong, the violation is useless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PlateRead:
    text: str
    confidence: float


class PlateOCR:
    def __init__(self, lang: str = "en", use_gpu: bool = True, fast_mode: bool = False):
        import easyocr
        self._reader = easyocr.Reader([lang], gpu=use_gpu, verbose=False)
        # fast_mode runs only 2 preprocessing variants (vs 6) — for CPU.
        self.fast_mode = fast_mode

    def read(self, plate_crop: np.ndarray) -> Optional[PlateRead]:
        if plate_crop is None or plate_crop.size == 0:
            return None

        candidates: list[PlateRead] = []
        for processed in self._preprocess_variants(plate_crop):
            result = self._ocr_once(processed)
            if result:
                candidates.append(result)

        if not candidates:
            return None
        return max(candidates, key=lambda r: self._score(r))

    def _ocr_once(self, img: np.ndarray) -> Optional[PlateRead]:
        try:
            # allowlist restricts EasyOCR to characters that appear on plates.
            result = self._reader.readtext(
                img,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
            )
        except Exception:
            return None

        if not result:
            return None

        texts: list[str] = []
        confs: list[float] = []
        for item in result:
            try:
                _, txt, conf = item
            except (ValueError, TypeError):
                continue
            txt = (txt or "").strip()
            if txt:
                texts.append(txt)
                confs.append(float(conf))

        if not texts:
            return None
        combined = "".join(texts).upper().replace(" ", "").replace("-", "")
        return PlateRead(text=combined, confidence=float(np.mean(confs)))

    def _score(self, read: PlateRead) -> float:
        score = read.confidence
        text = read.text
        alnum = sum(1 for c in text if c.isalnum())
        if 8 <= alnum <= 11:
            score += 0.15
        elif 6 <= alnum <= 12:
            score += 0.05
        if alnum < 5 or alnum > 14:
            score -= 0.3
        if len(text) >= 2 and text[0].isalpha() and text[1].isalpha():
            score += 0.1
        return score

    def _preprocess_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        variants = []
        crop = self._upscale(crop, min_height=80)
        crop = self._remove_border(crop)
        crop = self._deskew(crop)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        variants.append(crop.copy())
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        if self.fast_mode:
            return variants

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
        variants.append(cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR))

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))

        return variants

    @staticmethod
    def _upscale(crop: np.ndarray, min_height: int = 80) -> np.ndarray:
        h, w = crop.shape[:2]
        if h >= min_height:
            return crop
        scale = min_height / h
        new_w = int(w * scale)
        return cv2.resize(crop, (new_w, min_height), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _remove_border(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        margin_x = max(2, int(w * 0.05))
        margin_y = max(2, int(h * 0.08))
        trimmed = crop[margin_y:h - margin_y, margin_x:w - margin_x]
        if trimmed.size == 0:
            return crop
        return trimmed

    @staticmethod
    def _deskew(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h < 20 or w < 40:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=w // 3, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return crop

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:
                angles.append(angle)

        if not angles:
            return crop
        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return crop

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
