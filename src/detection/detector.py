"""Detection layer.

We run up to three Ultralytics models and fuse their outputs into one
canonical DetectionBundle:

  1. `general` — COCO-trained YOLO11. Vehicles, persons, traffic lights.
  2. `helmet`  — fine-tuned helmet/no-helmet head (optional). Two classes:
                 helmet, no_helmet, both annotated on the rider's head.
  3. `plate`   — fine-tuned license-plate head (optional). One class.

The helmet and plate models are loaded only if their weights file exists
on disk. This lets the system run end-to-end on day one with just the
base YOLO11 weights, while leaving clean slots to drop in better
domain-specific weights as they're trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from .classes import COCO_TO_CLASS, ObjectClass
from .plate_alpr import ALPRDetector


@dataclass
class Detection:
    cls: ObjectClass
    conf: float
    xyxy: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    text: Optional[str] = None           # pre-read plate text (from fast-alpr)
    text_conf: Optional[float] = None    # OCR confidence when text is pre-read

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionBundle:
    frame_idx: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)

    def of(self, cls: ObjectClass) -> list[Detection]:
        return [d for d in self.detections if d.cls is cls]

    def of_any(self, classes: set[ObjectClass]) -> list[Detection]:
        return [d for d in self.detections if d.cls in classes]


class Detector:
    def __init__(
        self,
        general_weights: Path,
        helmet_weights: Optional[Path] = None,
        plate_weights: Optional[Path] = None,
        device: str = "cuda:0",
        conf: float = 0.35,
        use_sahi_plate: bool = False,
        use_fast_alpr: bool = True,
    ):
        self.device = device
        self.conf = conf
        self.use_sahi_plate = use_sahi_plate
        self.general = YOLO(str(general_weights))

        self.helmet = None
        if helmet_weights and Path(helmet_weights).exists():
            self.helmet = YOLO(str(helmet_weights))

        # fast-alpr: try first; if unavailable, fall back to plate.pt
        self._alpr: Optional[ALPRDetector] = None
        if use_fast_alpr:
            alpr = ALPRDetector(device=device, conf=conf)
            if alpr.available:
                self._alpr = alpr

        self.plate = None
        if self._alpr is None:
            # Only load plate.pt when fast-alpr is not available
            if plate_weights and Path(plate_weights).exists():
                self.plate = YOLO(str(plate_weights))

        if self.use_sahi_plate and self.plate is None and self._alpr is None:
            self.use_sahi_plate = False

    def __call__(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> DetectionBundle:
        bundle = DetectionBundle(frame_idx=frame_idx, timestamp=timestamp)

        # --- general model ---
        res = self.general.predict(frame, device=self.device, conf=self.conf, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, confs, clses):
                cls = COCO_TO_CLASS.get(int(k))
                if cls is None:
                    continue
                bundle.detections.append(Detection(cls=cls, conf=float(c), xyxy=tuple(map(float, box))))

        # --- helmet head ---
        if self.helmet is not None:
            res = self.helmet.predict(frame, device=self.device, conf=self.conf, verbose=False)[0]
            names = res.names  # {0: 'helmet', 1: 'no_helmet'} or similar
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clses = res.boxes.cls.cpu().numpy().astype(int)
                for box, c, k in zip(xyxy, confs, clses):
                    label = names[int(k)].lower()
                    if ("no" in label or "without" in label) and "helmet" in label:
                        cls = ObjectClass.NO_HELMET
                    elif "helmet" in label:
                        cls = ObjectClass.HELMET
                    else:
                        continue
                    bundle.detections.append(Detection(cls=cls, conf=float(c), xyxy=tuple(map(float, box))))

        # --- plate head: fast-alpr (preferred) or legacy plate.pt ---
        if self._alpr is not None:
            for xyxy_box, text, text_c in self._alpr.detect(frame):
                bundle.detections.append(
                    Detection(
                        cls=ObjectClass.LICENSE_PLATE,
                        conf=text_c,
                        xyxy=xyxy_box,
                        text=text,
                        text_conf=text_c if text else None,
                    )
                )
        elif self.plate is not None:
            if self.use_sahi_plate:
                from .plate_sahi import detect_plates_sliced
                for xyxy_box, c in detect_plates_sliced(
                    frame, self.plate, self.device, self.conf
                ):
                    bundle.detections.append(
                        Detection(cls=ObjectClass.LICENSE_PLATE, conf=c, xyxy=xyxy_box)
                    )
            else:
                res = self.plate.predict(frame, device=self.device, conf=self.conf, verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    for box, c in zip(xyxy, confs):
                        bundle.detections.append(
                            Detection(
                                cls=ObjectClass.LICENSE_PLATE,
                                conf=float(c),
                                xyxy=tuple(map(float, box)),
                            )
                        )

        return bundle
