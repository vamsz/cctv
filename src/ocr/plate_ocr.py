"""License-plate OCR using PaddleOCR.

The detector gives us a plate bounding box; we crop it, optionally
upscale (PaddleOCR struggles below ~32 px text height), run OCR, and
combine the recognized line segments into a single plate string.

We return the *raw* OCR text and its mean confidence. Normalization to
the Indian plate format happens in plate_normalize.py and is called by
the rules engine, so this module stays a thin wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR


@dataclass
class PlateRead:
    text: str           # raw OCR concatenation
    confidence: float   # mean of per-token confidences, in [0, 1]


class PlateOCR:
    def __init__(self, lang: str = "en", use_gpu: bool = True):
        # use_angle_cls helps with slightly tilted plates.
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def read(self, plate_crop: np.ndarray) -> Optional[PlateRead]:
        if plate_crop is None or plate_crop.size == 0:
            return None
        # Upscale tiny crops; PaddleOCR's recognizer expects a min height.
        h, w = plate_crop.shape[:2]
        if h < 48:
            scale = 48 / h
            plate_crop = cv2.resize(plate_crop, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC)

        result = self._ocr.ocr(plate_crop, cls=True)
        if not result or not result[0]:
            return None

        texts: list[str] = []
        confs: list[float] = []
        for line in result[0]:
            # PaddleOCR returns [box, (text, conf)]
            try:
                _, (txt, conf) = line
            except (ValueError, TypeError):
                continue
            texts.append(txt)
            confs.append(float(conf))

        if not texts:
            return None
        return PlateRead(text="".join(texts), confidence=float(np.mean(confs)))
