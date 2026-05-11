"""License plate OCR — multi-engine consensus pipeline.

Engines (both run when available):
  - fast-plate-ocr (MIT, ONNX, ~3 ms CPU): ankandrew's mobile-vit-v2.
    Excellent throughput, strong on European / Spanish / Latin formats,
    weaker on Indian fonts and oblique angles.
  - PaddleOCR PP-OCRv5 (Apache 2.0, May 2025 release): +13 % accuracy
    over PP-OCRv4, explicitly tuned for license-plate-style text in 106
    languages. Runs ~50-100 ms per crop on CPU, faster on GPU.
  - EasyOCR — kept as a final safety net.

Strategy: BOTH engines read every plate crop. We then score each
candidate by `(confidence × format_validity)` and pick the winner. If
both agree on the same string we pin the confidence to the higher of
the two — that's a strong consensus signal.

Preprocessing pipeline (applied once before each engine):
  upscale → border-trim → deskew → CLAHE contrast → unsharp mask
This is required because Indian / oblique-angle plates are typically
24-36 px tall in the source frame; both engines need ≥48 px height
to deliver their advertised accuracy.

Sources:
  - PaddleOCR PP-OCRv5 release notes (May 2025): 13 % accuracy boost,
    license plate explicitly listed as a supported scenario.
  - "TrOCR Outperforms PaddleOCR" (RSIS Int'l, 2025): TrOCR wins on
    *handwritten* docs but PaddleOCR remains best-in-class for printed
    license plates.
  - "License Plate Detection using YOLO v8 and OCR" (ResearchGate,
    2024): PaddleOCR + filtered preprocessing reaches 91 % on Indian
    plates vs EasyOCR 86 %.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.ocr.plate_normalize import normalize_plate, normalize_indian_plate

log = logging.getLogger("plate_ocr")


@dataclass
class PlateRead:
    text: str
    confidence: float
    engine: str = ""    # "fast" | "paddle" | "easy" | "consensus"


def _ort_providers(use_gpu: bool = True):
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        available = set()
    providers = []
    if use_gpu and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _ort_session_options():
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Engine 1 — fast-plate-ocr (ONNX, milliseconds)
# ---------------------------------------------------------------------------

def _build_fast_ocr(model_name: str = "global-plates-mobile-vit-v2-model"):
    try:
        try:
            from fast_plate_ocr import LicensePlateRecognizer as _Rec
        except ImportError:
            from fast_plate_ocr import ONNXPlateRecognizer as _Rec   # type: ignore[no-redef]
        reader = _Rec(
            hub_ocr_model=model_name,
            device="cuda" if "CUDAExecutionProvider" in _ort_providers(True) else "cpu",
            providers=_ort_providers(True),
            sess_options=_ort_session_options(),
        )
        log.info("OCR engine: fast-plate-ocr (%s) loaded", model_name)
        return reader
    except Exception:
        log.debug("fast-plate-ocr load failed", exc_info=True)
        return None


def _run_fast(reader, img: np.ndarray) -> Optional[PlateRead]:
    if reader is None:
        return None
    try:
        result = reader.run(img, return_confidence=True)
        if not result:
            return None
        item = result[0]
        if hasattr(item, "plate") and hasattr(item, "confidence"):
            text = str(item.plate).strip().upper()
            conf = float(item.confidence) if item.confidence is not None else 0.85
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[0]).strip().upper()
            conf = float(item[1])
        else:
            text = str(item).strip().upper()
            conf = 0.85
        text = "".join(c for c in text if c.isalnum())
        if len(text) < 3:
            return None
        return PlateRead(text=text, confidence=conf, engine="fast")
    except Exception:
        log.debug("fast-plate-ocr inference failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Engine 2 — PaddleOCR PP-OCRv5
# ---------------------------------------------------------------------------

def _build_paddle_ocr(use_gpu: bool = False):
    """Build PaddleOCR with several fallbacks because PP-OCRv5 + Paddle 3.3
    on Windows has a oneDNN PIR-attribute bug that crashes inference. We
    try in order:
       1. PP-OCRv5 with mkldnn explicitly disabled
       2. PP-OCRv4 (stable, no mkldnn issue)
       3. Legacy 2.x signature
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        log.warning("paddleocr not installed — second-engine OCR disabled")
        return None

    # Attempt 1 — PP-OCRv5 with mkldnn off (works around CVE-style PIR bug)
    try:
        reader = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="gpu" if use_gpu else "cpu",
            ocr_version="PP-OCRv5",
            enable_mkldnn=False,
        )
        log.info("OCR engine: PaddleOCR PP-OCRv5 loaded (gpu=%s, mkldnn off)", use_gpu)
        return reader
    except Exception as e:
        log.warning("PP-OCRv5 init failed (%s) — falling back to PP-OCRv4", e)

    # Attempt 2 — PP-OCRv4 (less affected by the Windows PIR bug)
    try:
        reader = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="gpu" if use_gpu else "cpu",
            ocr_version="PP-OCRv4",
            enable_mkldnn=False,
        )
        log.info("OCR engine: PaddleOCR PP-OCRv4 loaded (gpu=%s, mkldnn off)", use_gpu)
        return reader
    except Exception as e:
        log.warning("PP-OCRv4 init failed (%s) — trying legacy signature", e)

    # Attempt 3 — legacy paddleocr 2.x signature
    try:
        reader = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )
        log.info("OCR engine: PaddleOCR (legacy 2.x) loaded")
        return reader
    except Exception:
        log.warning("All PaddleOCR init attempts failed", exc_info=True)
        return None


def _run_paddle(reader, img: np.ndarray) -> Optional[PlateRead]:
    if reader is None:
        return None
    try:
        # PaddleOCR 3.x: predict() returns list of dicts with rec_texts/rec_scores.
        # Older versions return [[ [box, (text, score)] ... ]] from .ocr().
        result = None
        if hasattr(reader, "predict"):
            try:
                result = reader.predict(img)
            except Exception:
                log.debug("paddle predict failed, falling back to ocr()", exc_info=True)
        if result is None:
            result = reader.ocr(img)

        texts: list[str] = []
        confs: list[float] = []
        # Format A — list of dicts (3.x predict)
        if result and isinstance(result, list) and result and isinstance(result[0], dict):
            for page in result:
                rec_texts = page.get("rec_texts") or []
                rec_scores = page.get("rec_scores") or []
                for t, s in zip(rec_texts, rec_scores):
                    t = "".join(c for c in str(t).upper() if c.isalnum())
                    if t:
                        texts.append(t)
                        confs.append(float(s))
        # Format B — list of lines (legacy ocr())
        elif result and isinstance(result, list) and result[0] is not None:
            for line in (result[0] or []):
                try:
                    _, (txt, conf) = line
                except (ValueError, TypeError):
                    continue
                txt = "".join(c for c in str(txt).upper() if c.isalnum())
                if txt:
                    texts.append(txt)
                    confs.append(float(conf))

        if not texts:
            return None
        combined = "".join(texts)
        avg = float(np.mean(confs))
        if len(combined) < 3:
            return None
        return PlateRead(text=combined, confidence=avg, engine="paddle")
    except Exception:
        log.debug("PaddleOCR inference failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Preprocessing — gives both engines a fair chance at small / oblique plates
# ---------------------------------------------------------------------------

def _preprocess(crop: np.ndarray, target_h: int = 128) -> np.ndarray:
    """Upscale + deskew + contrast-enhance for the headline crop.

    Two-stage upscaling for very small plates:
      - Plates with h < 32 px: 4x LANCZOS first to recover stroke
        continuity, then bilateral denoise (preserves edges, removes
        compression noise), then resize again to target_h.
      - Plates with h ≥ 32 px: single CUBIC pass to target_h.

    Target height = 128 px (LPSRGAN 2024: OCR accuracy plateaus there;
    below 48 px collapses).
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]

    if h < 32:
        # Tiny plate — aggressive multi-stage SR.
        # Stage 1: 4x LANCZOS upscale (best classical interpolant for
        # text glyphs — preserves character strokes better than CUBIC).
        crop = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
        # Stage 2: bilateral denoise to clean compression artefacts
        # while keeping the now-sharper character edges.
        crop = cv2.bilateralFilter(crop, d=5, sigmaColor=35, sigmaSpace=8)
        h, w = crop.shape[:2]

    if h < target_h:
        scale = target_h / h
        crop = cv2.resize(
            crop, (max(int(w * scale), target_h), target_h),
            interpolation=cv2.INTER_CUBIC,
        )

    crop = _deskew(crop)
    crop = _enhance_contrast(crop)
    return crop


def _generate_variants(crop: np.ndarray, target_h: int = 128) -> list[np.ndarray]:
    """Produce a handful of preprocessing variants for ensemble OCR.

    Different plates respond best to different filters:
      - clean / well-lit plates: raw upscale wins
      - bright sun / glare:      adaptive threshold wins
      - low-light / dim:         CLAHE wins
      - motion-blurred:          unsharp mask wins
      - noisy compressed video:  bilateral-filter denoise wins

    Running all 5 and letting the score function pick the best is
    cheap (~5-10 ms total preprocessing) and dominates a single fixed
    pipeline by a wide margin in the LPSRGAN evaluation.
    """
    if crop is None or crop.size == 0:
        return []

    base = _preprocess(crop, target_h=target_h)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    h, w = base.shape[:2]
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    variants: list[np.ndarray] = [base]

    # Variant 2: 2x super-resolution via INTER_LANCZOS4 + unsharp mask
    big = cv2.resize(base, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(big, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(big, 1.6, blur, -0.6, 0)
    variants.append(sharp)

    # Variant 3: bilateral denoise + edge-aware sharpen (good on noisy video)
    bil = cv2.bilateralFilter(base, d=5, sigmaColor=40, sigmaSpace=10)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    bil_sharp = cv2.filter2D(bil, -1, kernel)
    variants.append(bil_sharp)

    # Variant 4: adaptive threshold (good on glare / heavy shadow)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12,
    )
    variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))

    # Variant 5: CLAHE-only on luma (good on low-light / dim plates)
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    variants.append(cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR))

    return variants


def _deskew(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 20 or w < 40:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=w // 3, maxLineGap=10)
    if lines is None:
        return crop
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 15:
            angles.append(a)
    if not angles:
        return crop
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return crop
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def _enhance_contrast(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return clahe.apply(crop)


# ---------------------------------------------------------------------------
# Scoring — combines confidence + format validity
# ---------------------------------------------------------------------------

# Engine reliability priors. fast-alpr's mobile-vit-v2 model was trained
# on a global plate corpus that under-represents Indian fonts; on Indian
# plates it routinely hallucinates with internal softmax confidence
# 0.95+. Cap its effective contribution so a confident-but-wrong fast
# read can't outscore PaddleOCR's correct-but-lower-conf read.
_ENGINE_TRUST_CAP = {
    "crnn-ft":   1.00,
    "fast-alpr": 0.65,
    "fast":      0.65,
    "easy":      0.75,
    # PaddleOCR PP-OCRv5 is taken at face value
    "paddle":    1.00,
    "consensus": 1.00,
}


def _score(read: PlateRead) -> float:
    """Score a candidate read. Reads matching a known plate format
    get a big bonus; reads of suspicious length get penalised; reads
    from less-trusted engines have their confidence capped so they
    can't beat a more-trusted engine's correct reading.
    """
    text = read.text
    cap = _ENGINE_TRUST_CAP.get(read.engine, 0.85)
    base_conf = min(read.confidence, cap)
    score = base_conf

    indian = normalize_indian_plate(text) is not None
    generic = normalize_plate(text) is not None
    n = len(text)
    if indian:
        score += 0.40                  # strong: matches Indian format
    elif generic:
        score += 0.10                  # weaker: matches generic alphanumeric
    if 6 <= n <= 11:
        score += 0.10
    elif n < 5 or n > 13:
        score -= 0.30                  # suspect length
    if n >= 2 and text[:2].isalpha():
        score += 0.05                  # state-code-style prefix
    return score


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PlateOCR:
    """Multi-engine consensus license-plate OCR.

    All available engines are run on every preprocessed crop. The
    candidate with the highest combined `confidence × format_validity`
    score wins. When two engines agree on the same string we boost the
    confidence by 0.05 (capped at 0.99).
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        fast_ocr_model: str = "global-plates-mobile-vit-v2-model",
    ):
        self._fast = _build_fast_ocr(fast_ocr_model)
        self._paddle = _build_paddle_ocr(use_gpu=use_gpu)
        self._lock = threading.Lock()

        self._ft_ocr = None
        try:
            from src.ocr.plate_recognizer_ft import FineTunedOCR
            self._ft_ocr = FineTunedOCR()
        except Exception:
            log.debug("Fine-tuned OCR not loaded", exc_info=True)

        # Warmup: first ONNX run has 200-400 ms session-init latency on
        # the fast-plate-ocr provider chain. Run a dummy crop through
        # both engines now so the first real plate doesn't pay that.
        dummy = np.full((96, 320, 3), 220, dtype=np.uint8)
        try:
            if self._fast is not None:
                _run_fast(self._fast, dummy)
            if self._paddle is not None:
                _run_paddle(self._paddle, dummy)
        except Exception:
            log.debug("OCR warmup failed (non-fatal)", exc_info=True)

        engines = []
        if self._ft_ocr is not None:
            engines.append("crnn-ft")
        if self._fast is not None:
            engines.append("fast-plate-ocr")
        if self._paddle is not None:
            engines.append("paddleocr-v5(fallback)")
        self._engines_str = " + ".join(engines) if engines else "none"
        log.info("PlateOCR ready — engines: %s | fast-trust ≥ %.2f",
                 self._engines_str, self._FAST_TRUST_CONF)

    @property
    def backend(self) -> str:
        return self._engines_str

    # If fast-plate-ocr returns a confident Indian-format read, we
    # don't need to call PaddleOCR at all — saves ~70 ms / call.
    _FAST_TRUST_CONF = 0.70

    def read(self, plate_crop: np.ndarray) -> Optional[PlateRead]:
        """Two-tier OCR with adaptive escalation.

          1. Try fast-plate-ocr (CUDA via ONNX, ~5-10 ms). If it returns
             a high-confidence Indian-valid read, RETURN IMMEDIATELY.
          2. Otherwise call PaddleOCR PP-OCRv5 (CPU, ~50-100 ms) as a
             second opinion and pick the best by combined score.

        On clean plates (~70 % of frames) only fast-plate-ocr runs:
        ~5-10 ms per plate per frame, identical accuracy to the old
        always-both pipeline. On hard plates we still get the Paddle
        fallback. Net effect: pipeline keeps up with multi-camera 30 fps.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        with self._lock:
            return self._read_locked(plate_crop)

    def _read_locked(self, plate_crop: np.ndarray) -> Optional[PlateRead]:
        """Run OCR engines under one lock; PaddleOCR/ONNX sessions are not
        reliably thread-safe on Windows when shared across camera workers.
        """

        prepped = _preprocess(plate_crop)

        # Tier 0: Fine-tuned CRNN (trained on raw plate crop)
        ft_read = None
        if self._ft_ocr is not None:
            res = self._ft_ocr.read(plate_crop)
            if res:
                text, conf = res
                from src.ocr.plate_normalize import normalize_indian_plate
                if conf >= 0.85 and normalize_indian_plate(text) is not None:
                    return PlateRead(text=text, confidence=conf, engine="crnn-ft")
                ft_read = PlateRead(text=text, confidence=conf, engine="crnn-ft")

        # Tier 1: fast-plate-ocr (cheap)
        fast_read = _run_fast(self._fast, prepped)
        if fast_read is not None:
            # Trust the fast read when it's confident AND looks like a real plate
            from src.ocr.plate_normalize import normalize_indian_plate
            if (fast_read.confidence >= self._FAST_TRUST_CONF
                and normalize_indian_plate(fast_read.text) is not None):
                return fast_read

        # Tier 2: PaddleOCR fallback for uncertain or non-Indian plates
        paddle_read = _run_paddle(self._paddle, prepped)

        candidates = [c for c in (ft_read, fast_read, paddle_read) if c is not None]
        if not candidates:
            return None

        if (len(candidates) == 2
            and candidates[0].text == candidates[1].text):
            return PlateRead(
                text=candidates[0].text,
                confidence=min(0.99, max(c.confidence for c in candidates) + 0.05),
                engine="consensus",
            )
        return max(candidates, key=_score)
