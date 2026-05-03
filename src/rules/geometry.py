"""Small geometry helpers used across rules."""
from __future__ import annotations


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> float:
    """Fraction of `inner`'s area that lies within `outer`."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    cx1, cy1 = max(ox1, ix1), max(oy1, iy1)
    cx2, cy2 = min(ox2, ix2), min(oy2, iy2)
    cw, ch = max(0.0, cx2 - cx1), max(0.0, cy2 - cy1)
    inter = cw * ch
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / inner_area if inner_area > 0 else 0.0


def signed_distance_to_line(point: tuple[float, float], line_p1: tuple[float, float], line_p2: tuple[float, float]) -> float:
    """Signed perpendicular distance from `point` to the infinite line through line_p1->line_p2.

    Sign tells you which side of the line the point is on, which is how
    the red-light rule decides "before" vs "after" the stop line.
    """
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
