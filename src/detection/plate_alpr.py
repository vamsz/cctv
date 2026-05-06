"""fast-alpr end-to-end license plate detector + OCR.

Wraps ankandrew/fast-alpr (MIT, pip install fast-alpr) which bundles:
  - yolo-v9-t-384-license-plate-end2end  (ONNX plate detector)
  - global-plates-mobile-vit-v2-model    (ONNX OCR, 93.3% global accuracy)

Returns list[(xyxy, text, confidence)] in a single forward pass.
Falls back gracefully to an empty list when the package is not installed
so the legacy plate.pt / EasyOCR path continues to work.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("plate_alpr")

_ALPR_AVAILABLE = False

try:
    from fast_alpr import ALPR as _FastALPR      # pip install fast-alpr
    _ALPR_AVAILABLE = True
except ImportError:
    pass


def _bbox_to_xyxy(bbox) -> tuple[float, float, float, float]:
    """Normalise any bounding-box format from fast-alpr to (x1,y1,x2,y2)."""
    # BoundingBox dataclass with x1/y1/x2/y2 attributes
    if hasattr(bbox, "x1") and hasattr(bbox, "y1"):
        return (float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2))
    # BoundingBox as (x, y, w, h) tuple
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        a, b, c, d = (float(v) for v in bbox)
        # Distinguish xyxy from xywh: if c > a and d > b, likely xyxy
        if c > a and d > b and (c - a) < 2000 and (d - b) < 2000:
            return (a, b, c, d)
        # Otherwise treat as x, y, w, h
        return (a, b, a + c, b + d)
    return tuple(map(float, bbox))  # type: ignore[return-value]


class ALPRDetector:
    """End-to-end license plate detector using fast-alpr.

    Usage:
        det = ALPRDetector(detector_model="yolo-v9-t-384-license-plate-end2end",
                           ocr_model="global-plates-mobile-vit-v2-model")
        plates = det.detect(frame)
        # → [(x1,y1,x2,y2, text_or_None, confidence), ...]
    """

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "global-plates-mobile-vit-v2-model",
        device: str = "cpu",
        conf: float = 0.25,
    ):
        self._available = _ALPR_AVAILABLE
        self._alpr = None
        if not _ALPR_AVAILABLE:
            log.warning(
                "fast-alpr not installed — plate detection falls back to plate.pt/EasyOCR. "
                "Run: pip install fast-alpr fast-plate-ocr"
            )
            return
        try:
            # fast-alpr auto-downloads ONNX weights on first use
            self._alpr = _FastALPR(
                detector_model=detector_model,
                ocr_model=ocr_model,
            )
            log.info("fast-alpr loaded: detector=%s  ocr=%s", detector_model, ocr_model)
        except Exception:
            log.exception("fast-alpr init failed — falling back to legacy pipeline")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._alpr is not None

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[tuple[tuple[float, float, float, float], Optional[str], float]]:
        """
        Returns list of (xyxy_pixels, text_or_None, confidence).
        Returns empty list when fast-alpr is unavailable.
        """
        if not self.available or self._alpr is None:
            return []
        try:
            results = self._alpr.predict(frame)
            out = []
            for r in results:
                try:
                    xyxy = _bbox_to_xyxy(r.detection.bounding_box)
                    det_conf = float(r.detection.confidence)
                    text = None
                    text_conf = det_conf
                    if r.ocr is not None:
                        text = str(r.ocr.text).strip().upper().replace(" ", "")
                        text_conf = float(r.ocr.confidence)
                    out.append((xyxy, text, text_conf))
                except Exception:
                    pass
            return out
        except Exception:
            log.debug("fast-alpr inference failed", exc_info=True)
            return []
