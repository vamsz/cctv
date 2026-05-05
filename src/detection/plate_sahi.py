"""Sliced plate detection (manual SAHI-style).

Divides the frame into overlapping tiles, runs the plate model on each
tile, transforms detections back to frame coordinates, then NMS-merges
the results. This recovers plates that a single-pass detector misses on
distant/small vehicles.

Only activated for vehicles whose bounding-box area is below a threshold
(close vehicles already produce large, readable plates with single-pass).

Tiles: 2×2 grid with 20% overlap → 4 inference calls per frame.
GPU latency for the plate model: ~3 ms per tile → +9 ms worst case.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[tuple], confs: list[float], iou_thresh: float = 0.5) -> list[int]:
    if not boxes:
        return []
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j != i and j not in suppressed and _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)
    return keep


def detect_plates_sliced(
    frame: np.ndarray,
    plate_model,
    device: str,
    conf: float,
    slice_rows: int = 2,
    slice_cols: int = 2,
    overlap: float = 0.20,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """
    Returns list of (xyxy, confidence) in full-frame coordinates.
    plate_model must be a callable YOLO model (same API as in Detector).
    """
    H, W = frame.shape[:2]
    stride_y = int(H / (slice_rows - overlap * (slice_rows - 1)))
    stride_x = int(W / (slice_cols - overlap * (slice_cols - 1)))
    tile_h = int(stride_y * (1 + overlap))
    tile_w = int(stride_x * (1 + overlap))

    all_boxes: list[tuple] = []
    all_confs: list[float] = []

    for row in range(slice_rows):
        for col in range(slice_cols):
            y0 = min(row * stride_y, H - tile_h)
            x0 = min(col * stride_x, W - tile_w)
            y1 = min(y0 + tile_h, H)
            x1 = min(x0 + tile_w, W)
            tile = frame[y0:y1, x0:x1]

            res = plate_model.predict(tile, device=device, conf=conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue

            xyxy_tile = res.boxes.xyxy.cpu().numpy()
            confs_tile = res.boxes.conf.cpu().numpy()
            for box, c in zip(xyxy_tile, confs_tile):
                # transform to full frame coords
                fx1, fy1, fx2, fy2 = (
                    float(box[0]) + x0,
                    float(box[1]) + y0,
                    float(box[2]) + x0,
                    float(box[3]) + y0,
                )
                all_boxes.append((fx1, fy1, fx2, fy2))
                all_confs.append(float(c))

    keep = _nms(all_boxes, all_confs, iou_thresh=0.45)
    return [(all_boxes[i], all_confs[i]) for i in keep]
