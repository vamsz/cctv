"""Real face detector using YOLOv11n-face from HuggingFace.

Backbone: AdamCodd/YOLOv11n-face-detection
  WIDERFACE-trained, 225 epochs.
  AP — Easy: 0.942, Medium: 0.921, Hard: 0.810
  ~2.6 M parameters, ~3-5 ms per frame on RTX 3070.

This replaces the upper-30%-of-person-bbox heuristic. The old heuristic
captured legs / torsos when the camera caught people sitting, lying,
or at oblique angles — making the police-DB embedding match against
clothing rather than face geometry.

Reference: https://huggingface.co/AdamCodd/YOLOv11n-face-detection
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("face.yolo")

_HF_HUB_AVAILABLE = False
try:
    from huggingface_hub import hf_hub_download
    _HF_HUB_AVAILABLE = True
except ImportError:
    pass


@dataclass
class FaceCrop:
    xyxy: tuple[float, float, float, float]
    image: np.ndarray
    quality: float
    person_track_id: Optional[int] = None
    detection_conf: float = 0.0


class YoloFaceDetector:
    """YOLOv11n face detector with optional person-bbox restriction.

    Two-stage:
      1. Run YOLOv11n-face on the FULL frame → list of face bboxes.
      2. Filter to faces whose centre lies inside a tracked person bbox
         (so we only capture *participants* in the active incident, not
         passers-by or onlookers reflected in shop windows).

    Also computes a sharpness score so blurred-face captures (most
    common failure for embedding match) are dropped before they reach
    the police DB.
    """

    REPO_ID = "AdamCodd/YOLOv11n-face-detection"
    MODEL_FILENAME = "model.pt"

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.35,
        min_side_px: int = 24,
        min_quality: float = 10.0,
        enhance_crops: bool = True,
        min_output_side: int = 112,
        context_pad: float = 0.55,
        max_upscale: float = 2.5,
        require_inside_person: bool = True,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.min_side_px = min_side_px
        self.min_quality = min_quality
        self.require_inside_person = require_inside_person
        self.enhance_crops = enhance_crops
        self.min_output_side = min_output_side
        self.context_pad = context_pad
        self.max_upscale = max_upscale
        self._model = None
        self._available = _HF_HUB_AVAILABLE
        if not self._available:
            log.warning("huggingface_hub not installed — YoloFaceDetector disabled")

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            log.info("Downloading face model %s ...", self.REPO_ID)
            path = hf_hub_download(repo_id=self.REPO_ID, filename=self.MODEL_FILENAME)
            self._model = YOLO(path)
            # Warmup so the first real predict doesn't crash on meta-tensors
            try:
                self._model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                                    device=self.device, verbose=False,
                                    half=self.device.startswith("cuda"))
            except Exception:
                log.debug("face model warmup failed", exc_info=True)
            log.info("YOLOv11n face detector loaded on %s", self.device)
        except Exception:
            log.exception("YoloFaceDetector load failed — disabling")
            self._available = False

    def extract_faces(
        self,
        frame: np.ndarray,
        person_bboxes: list[tuple[float, float, float, float]],
        track_ids: Optional[list[Optional[int]]] = None,
    ) -> list[FaceCrop]:
        if not self._available:
            return []
        self._ensure_loaded()
        if not self._available or self._model is None:
            return []

        try:
            results = self._model.predict(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False,
                half=self.device.startswith("cuda"),
            )
        except Exception:
            log.debug("YOLO face predict failed", exc_info=True)
            return []

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        h, w = frame.shape[:2]

        out: list[FaceCrop] = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < self.min_side_px or y2 - y1 < self.min_side_px:
                continue

            # Restrict to faces inside an active person bbox if requested
            tid = None
            if self.require_inside_person and person_bboxes:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                tid = self._inside_which_person(cx, cy, person_bboxes, track_ids)
                if tid is None:
                    continue

            tight_crop = frame[y1:y2, x1:x2]
            if tight_crop.size == 0:
                continue
            quality = _laplacian_variance(tight_crop)
            if quality < self.min_quality:
                continue

            # Save a padded face/head crop rather than an aggressively zoomed
            # tight face. Operators need context, and huge upscales turn
            # 24px CCTV faces into unreadable blocky thumbnails.
            cx1, cy1, cx2, cy2 = _expand_box(
                (x1, y1, x2, y2), w, h, pad=self.context_pad,
            )
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            enhanced = crop.copy()
            if self.enhance_crops:
                enhanced = _enhance_face_crop(
                    enhanced, self.min_output_side, max_upscale=self.max_upscale,
                )

            out.append(FaceCrop(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                image=enhanced,
                quality=float(quality),
                person_track_id=tid,
                detection_conf=float(conf),
            ))

        return out

    @staticmethod
    def _inside_which_person(
        cx: float, cy: float,
        person_bboxes: list[tuple[float, float, float, float]],
        track_ids: Optional[list[Optional[int]]],
    ) -> Optional[int]:
        for i, (px1, py1, px2, py2) in enumerate(person_bboxes):
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if track_ids is not None and i < len(track_ids):
                    return track_ids[i]
                return -1     # inside someone but no track id
        return None


def _laplacian_variance(img: np.ndarray) -> float:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _expand_box(
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    pad: float = 0.55,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad)
    py_top = int(bh * pad)
    py_bottom = int(bh * pad * 1.4)
    return (
        max(0, x1 - px),
        max(0, y1 - py_top),
        min(frame_w, x2 + px),
        min(frame_h, y2 + py_bottom),
    )


def _enhance_face_crop(
    crop: np.ndarray,
    min_output_side: int = 160,
    max_upscale: float = 3.0,
) -> np.ndarray:
    """Upscale small faces and apply CLAHE to improve visibility.

    CCTV faces are typically 30-60px tall — too small and dark for human
    review or embedding models. This:
      1. Upscales to at least `min_output_side` using bicubic interpolation
      2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
         to brighten dark faces without blowing out highlights
      3. Applies a gentle unsharp mask to recover edge detail lost in upscale
    """
    h, w = crop.shape[:2]
    # 1) Upscale if either dimension is below minimum
    if h < min_output_side or w < min_output_side:
        scale = min(max(min_output_side / h, min_output_side / w), max_upscale)
        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_LANCZOS4 if scale <= 2.0 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    # 2) CLAHE on the L channel of LAB colour space
    if crop.ndim == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l_ch = clahe.apply(l_ch)
        crop = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    # 3) Gentle unsharp mask — sharpen without amplifying noise
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=2.0)
    crop = cv2.addWeighted(crop, 1.3, blurred, -0.3, 0)

    return crop
