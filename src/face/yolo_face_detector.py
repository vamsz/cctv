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
        min_side_px: int = 36,
        min_quality: float = 18.0,
        require_inside_person: bool = True,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.min_side_px = min_side_px
        self.min_quality = min_quality
        self.require_inside_person = require_inside_person
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

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            quality = _laplacian_variance(crop)
            if quality < self.min_quality:
                continue

            out.append(FaceCrop(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                image=crop.copy(),
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
