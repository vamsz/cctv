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

import logging

from .classes import COCO_TO_CLASS, ObjectClass
from .plate_alpr import ALPRDetector

log = logging.getLogger("detection.detector")


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _dedupe_plates(plates, iou_thresh: float = 0.4):
    """Merge overlapping plate detections from multiple detectors.

    When two boxes overlap above the threshold we keep the one with
    text (fast-alpr's bundled OCR run) over the bare bbox detection.
    Otherwise highest confidence wins.
    """
    kept = []
    for det in plates:
        # find an overlapping kept detection
        merged = False
        for i, k in enumerate(kept):
            if _iou(det.xyxy, k.xyxy) >= iou_thresh:
                # one with text wins; otherwise higher confidence wins
                if det.text and not k.text:
                    kept[i] = det
                elif k.text and not det.text:
                    pass
                elif det.conf > k.conf:
                    kept[i] = det
                merged = True
                break
        if not merged:
            kept.append(det)
    return kept


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
        self._half = device.startswith("cuda")  # fp16 — CUDA only, ~1.5-2x faster
        self.general = YOLO(str(general_weights))
        self._warmup(self.general, "general")

        self.helmet = None
        if helmet_weights and Path(helmet_weights).exists():
            # Auto-upgrade: prefer fine-tuned helmet_ft.pt when it exists next to helmet.pt
            ft = Path(helmet_weights).parent / "helmet_ft.pt"
            if ft.exists():
                helmet_weights = ft
            self.helmet = YOLO(str(helmet_weights))
            self._warmup(self.helmet, "helmet")
            log.info("Helmet model: %s", Path(helmet_weights).name)

        # fast-alpr (global) detector — kept available for foreign plates,
        # not just gated behind use_fast_alpr=True. We always run BOTH
        # detectors when both exist so that:
        #   - plate_ft.pt catches Indian plates well
        #   - fast-alpr catches everything else (EU / NA / etc.)
        # and we IoU-dedupe at the end.
        self._alpr: Optional[ALPRDetector] = None
        if use_fast_alpr:
            alpr = ALPRDetector(device=device, conf=conf)
            if alpr.available:
                self._alpr = alpr
                log.info("Plate detector: fast-alpr (global)")

        # plate.pt / plate_ft.pt — fine-tuned domain detector
        self.plate = None
        if plate_weights and Path(plate_weights).exists():
            pt_ft = Path(plate_weights).parent / "plate_ft.pt"
            if pt_ft.exists():
                plate_weights = pt_ft
            self.plate = YOLO(str(plate_weights))
            self._warmup(self.plate, "plate")
            log.info("Plate detector: %s", Path(plate_weights).name)

        # Make sure we have *some* plate detector
        if self.plate is None and self._alpr is None and not use_fast_alpr:
            # last resort: spin up fast-alpr even if user disabled it
            alpr = ALPRDetector(device=device, conf=conf)
            if alpr.available:
                self._alpr = alpr
                log.info("Plate detector fallback: fast-alpr (global)")

        if self.use_sahi_plate and self.plate is None and self._alpr is None:
            self.use_sahi_plate = False

    def _warmup(self, model: YOLO, name: str) -> None:
        """Run a single dummy predict so model weights are fully materialized
        on the target device. With torch >= 2.5 + ultralytics 8.3.x, plain
        ``model.to(device)`` leaves params on a meta tensor — the first real
        predict (especially batched / with half=True) then crashes inside
        ``model.fuse()``. Warming up here avoids that completely.
        """
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy, device=self.device, verbose=False, half=self._half)
        except Exception:
            log.exception("YOLO warmup failed for %s — first real predict may crash", name)

    def _parse_helmet_results(self, res) -> list[Detection]:
        names = res.names
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy_arr = res.boxes.xyxy.cpu().numpy()
        confs_arr = res.boxes.conf.cpu().numpy()
        clses_arr = res.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy_arr, confs_arr, clses_arr):
            label = names[int(k)].lower()
            if ("no" in label or "without" in label) and "helmet" in label:
                cls = ObjectClass.NO_HELMET
            elif "helmet" in label:
                cls = ObjectClass.HELMET
            else:
                continue
            out.append(Detection(cls=cls, conf=float(c), xyxy=tuple(map(float, box))))
        return out

    def _helmet_full_frame(self, frame: np.ndarray) -> list[Detection]:
        res = self.helmet.predict(frame, device=self.device, conf=self.conf, verbose=False, half=self._half)[0]
        return self._parse_helmet_results(res)

    def _helmet_two_stage(self, frame: np.ndarray, riders: list[Detection]) -> list[Detection]:
        """Crop the upper 45 % of each rider bbox and run the helmet model in one batched call.

        Rationale: the head occupies the top portion of a person/bike bounding box.
        Isolating it at full crop resolution dramatically boosts recall for small helmets
        and reduces false positives from unrelated objects elsewhere in the frame.
        All crops are forwarded in a single batched GPU call for efficiency.
        """
        h_frame, w_frame = frame.shape[:2]

        # Collect valid crops + their offsets for coordinate remapping
        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []   # (cx1, cy1) per crop

        for rider in riders:
            rx1, ry1, rx2, ry2 = rider.xyxy
            # Upper 45 % covers head + shoulders for both standing persons and seated riders
            cy2_raw = ry1 + (ry2 - ry1) * 0.45
            cx1 = max(0, int(rx1))
            cy1 = max(0, int(ry1))
            cx2 = min(w_frame, int(rx2))
            cy2 = min(h_frame, int(cy2_raw))

            if cx2 - cx1 < 24 or cy2 - cy1 < 24:
                continue  # crop too small — skip

            crops.append(frame[cy1:cy2, cx1:cx2])
            offsets.append((cx1, cy1))

        if not crops:
            return []

        # Single batched inference — YOLO accepts a list of images
        results = self.helmet.predict(crops, device=self.device, conf=self.conf, verbose=False, half=self._half)

        out: list[Detection] = []
        seen: set[tuple] = set()  # deduplicate by (cls, rounded_box)

        for res, (ox, oy) in zip(results, offsets):
            for det in self._parse_helmet_results(res):
                bx1, by1, bx2, by2 = det.xyxy
                fx1, fy1 = float(bx1) + ox, float(by1) + oy
                fx2, fy2 = float(bx2) + ox, float(by2) + oy
                key = (det.cls, round(fx1), round(fy1), round(fx2), round(fy2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Detection(cls=det.cls, conf=det.conf, xyxy=(fx1, fy1, fx2, fy2)))

        return out

    def __call__(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> DetectionBundle:
        bundle = DetectionBundle(frame_idx=frame_idx, timestamp=timestamp)

        # --- general model ---
        res = self.general.predict(
            frame,
            device=self.device,
            conf=self.conf,
            classes=list(COCO_TO_CLASS.keys()),
            verbose=False,
            half=self._half,
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, confs, clses):
                cls = COCO_TO_CLASS.get(int(k))
                if cls is None:
                    continue
                bundle.detections.append(Detection(cls=cls, conf=float(c), xyxy=tuple(map(float, box))))

        # --- helmet head: two-stage (crop upper-body of each rider, then classify) ---
        if self.helmet is not None:
            riders = bundle.of_any({ObjectClass.TWO_WHEELER, ObjectClass.PERSON})
            if riders:
                bundle.detections.extend(self._helmet_two_stage(frame, riders))
            else:
                # No riders detected — fall back to full-frame pass
                bundle.detections.extend(self._helmet_full_frame(frame))

        # --- plate head: run BOTH detectors when available, IoU-dedupe ---
        plate_dets: list[Detection] = []

        if self._alpr is not None:
            for xyxy_box, text, text_c in self._alpr.detect(frame):
                plate_dets.append(Detection(
                    cls=ObjectClass.LICENSE_PLATE,
                    conf=text_c,
                    xyxy=xyxy_box,
                    text=text,
                    text_conf=text_c if text else None,
                ))

        if self.plate is not None:
            res = self.plate.predict(
                frame, device=self.device, conf=self.conf,
                verbose=False, half=self._half,
            )[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for box, c in zip(xyxy, confs):
                    plate_dets.append(Detection(
                        cls=ObjectClass.LICENSE_PLATE,
                        conf=float(c),
                        xyxy=tuple(map(float, box)),
                    ))

        # IoU-dedupe overlapping plate detections from the two detectors.
        # When both fired on the same plate we prefer the fast-alpr one
        # because it carries the bundled OCR text already.
        bundle.detections.extend(_dedupe_plates(plate_dets, iou_thresh=0.4))

        return bundle
