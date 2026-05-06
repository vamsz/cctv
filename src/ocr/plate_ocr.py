"""License plate OCR — tiered backend selection.

Priority (auto-detected at init):
  1. fast-plate-ocr (MIT, pip install fast-plate-ocr)
       global-plates-mobile-vit-v2-model: 2.9 ms / 344 plates-per-second on M1 CPU;
       cct-xs-v2-global-model: 0.47 ms RTX 3090 (TensorRT); 93.3% global accuracy.
       Drop from 200-400 ms (EasyOCR) to ~3-10 ms CPU, ~1 ms GPU.
  2. PaddleOCR ≥ 3.0.3 (Apache 2.0) — Windows MKLDNN fix confirmed in 3.0.3 changelog.
       PP-OCRv5 mobile: 370+ chars/sec on Intel Xeon Gold with MKLDNN.
  3. EasyOCR (Apache 2.0) — legacy; always available; slowest (200-400 ms CPU).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("plate_ocr")


@dataclass
class PlateRead:
    text: str
    confidence: float


# ---------------------------------------------------------------------------
# Backend 1: fast-plate-ocr
# ---------------------------------------------------------------------------

def _try_build_fast_ocr(model_name: str = "global-plates-mobile-vit-v2-model"):
    try:
        # v0.3+ uses LicensePlateRecognizer; older used ONNXPlateRecognizer
        try:
            from fast_plate_ocr import LicensePlateRecognizer as _Rec
        except ImportError:
            from fast_plate_ocr import ONNXPlateRecognizer as _Rec   # type: ignore[no-redef]
        reader = _Rec(hub_ocr_model=model_name)
        log.info("PlateOCR backend: fast-plate-ocr (%s)", model_name)
        return reader, "fast"
    except ImportError:
        return None, None
    except Exception as exc:
        log.warning("fast-plate-ocr init failed (%s) — trying PaddleOCR", exc)
        return None, None


def _fast_read(reader, img: np.ndarray) -> Optional[PlateRead]:
    """Run fast-plate-ocr on a single BGR crop."""
    try:
        result = reader.run(img, return_confidence=True)
        # v0.3+: returns List[PlatePrediction] where each has .plate and .confidence
        # older: List[str] or List[Tuple[str, float]]
        if not result:
            return None
        item = result[0]
        # PlatePrediction dataclass
        if hasattr(item, "plate") and hasattr(item, "confidence"):
            text = str(item.plate).strip().upper().replace(" ", "").replace("-", "")
            conf = float(item.confidence) if item.confidence is not None else 0.85
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[0]).strip().upper().replace(" ", "").replace("-", "")
            conf = float(item[1])
        else:
            text = str(item).strip().upper().replace(" ", "").replace("-", "")
            conf = 0.85
        if len(text) < 3:
            return None
        return PlateRead(text=text, confidence=conf)
    except Exception:
        log.debug("fast-plate-ocr read failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Backend 2: PaddleOCR ≥ 3.0.3
# ---------------------------------------------------------------------------

def _try_build_paddle_ocr():
    try:
        from paddleocr import PaddleOCR  # pip install paddlepaddle paddleocr>=3.0.3
        reader = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            enable_mkldnn=True,    # Windows MKLDNN fix: valid in ≥ 3.0.3
            show_log=False,
        )
        log.info("PlateOCR backend: PaddleOCR (PP-OCRv5)")
        return reader, "paddle"
    except ImportError:
        return None, None
    except Exception as exc:
        log.warning("PaddleOCR init failed (%s) — using EasyOCR", exc)
        return None, None


def _paddle_read(reader, img: np.ndarray) -> Optional[PlateRead]:
    try:
        result = reader.ocr(img, cls=False)
        if not result or result == [None]:
            return None
        texts, confs = [], []
        for line in (result[0] or []):
            # line = [[pts], [text, conf]]
            try:
                _, (txt, conf) = line
                txt = str(txt).strip().upper().replace(" ", "").replace("-", "")
                if txt:
                    texts.append(txt)
                    confs.append(float(conf))
            except Exception:
                continue
        if not texts:
            return None
        combined = "".join(texts)
        return PlateRead(text=combined, confidence=float(np.mean(confs)))
    except Exception:
        log.debug("PaddleOCR read failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Backend 3: EasyOCR (legacy fallback)
# ---------------------------------------------------------------------------

def _try_build_easyocr(lang: str = "en", use_gpu: bool = False):
    import easyocr
    reader = easyocr.Reader([lang], gpu=use_gpu, verbose=False)
    log.info("PlateOCR backend: EasyOCR (legacy fallback)")
    return reader, "easy"


def _easyocr_read(reader, img: np.ndarray) -> Optional[PlateRead]:
    try:
        result = reader.readtext(
            img,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
        )
    except Exception:
        return None
    if not result:
        return None
    texts, confs = [], []
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


# ---------------------------------------------------------------------------
# Preprocessing helpers (shared by all backends)
# ---------------------------------------------------------------------------

def _preprocess_crop(crop: np.ndarray) -> list[np.ndarray]:
    """Return 6 preprocessed variants of a plate crop for ensemble OCR."""
    crop = _upscale(crop, min_height=80)
    crop = _remove_border(crop)
    crop = _deskew(crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    variants = [
        crop.copy(),
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
    ]

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR))

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))

    return variants


def _score(read: PlateRead) -> float:
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


def _upscale(crop: np.ndarray, min_height: int = 80) -> np.ndarray:
    h, w = crop.shape[:2]
    if h >= min_height:
        return crop
    scale = min_height / h
    return cv2.resize(crop, (int(w * scale), min_height), interpolation=cv2.INTER_CUBIC)


def _remove_border(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    mx, my = max(2, int(w * 0.05)), max(2, int(h * 0.08))
    trimmed = crop[my:h - my, mx:w - mx]
    return trimmed if trimmed.size > 0 else crop


def _deskew(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 20 or w < 40:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=w // 3, maxLineGap=10)
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
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return crop
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# PlateOCR — public interface (unchanged from caller perspective)
# ---------------------------------------------------------------------------

class PlateOCR:
    """Multi-backend license plate OCR with automatic backend selection.

    The same `read(crop) → PlateRead | None` API regardless of backend.
    fast-plate-ocr runs a single variant (already optimised for plates).
    PaddleOCR and EasyOCR run all 6 preprocessing variants and pick the best.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        fast_mode: bool = False,
        fast_ocr_model: str = "global-plates-mobile-vit-v2-model",
    ):
        self._backend = None
        self._reader = None

        # Priority 1: fast-plate-ocr
        reader, backend = _try_build_fast_ocr(fast_ocr_model)
        if reader is not None:
            self._reader, self._backend = reader, backend
            return

        # Priority 2: PaddleOCR ≥ 3.0.3
        reader, backend = _try_build_paddle_ocr()
        if reader is not None:
            self._reader, self._backend = reader, backend
            return

        # Priority 3: EasyOCR
        reader, backend = _try_build_easyocr(lang, use_gpu)
        self._reader, self._backend = reader, backend

        self._fast_mode = fast_mode

    @property
    def backend(self) -> str:
        return self._backend or "none"

    def read(self, plate_crop: np.ndarray) -> Optional[PlateRead]:
        if plate_crop is None or plate_crop.size == 0:
            return None

        # fast-plate-ocr: single pass on original crop (model handles sizing internally)
        if self._backend == "fast":
            return _fast_read(self._reader, plate_crop)

        # PaddleOCR / EasyOCR: multi-variant ensemble
        read_fn = _paddle_read if self._backend == "paddle" else _easyocr_read
        candidates: list[PlateRead] = []
        for variant in _preprocess_crop(plate_crop):
            r = read_fn(self._reader, variant)
            if r:
                candidates.append(r)

        if not candidates:
            return None
        return max(candidates, key=_score)
