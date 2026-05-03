"""Prometheus metrics, exposed by both the pipeline and the API.

Naming follows Prometheus conventions:
  cctv_<subsystem>_<metric>_<unit>

Pipeline-side metrics drive ops dashboards (frames/sec per camera,
inference latency, OCR success rate, violations fired). API-side
metrics drive the operator workload dashboard (review backlog age,
approval rate).
"""
from __future__ import annotations

import threading
import time

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---- pipeline-side
frames_total = Counter(
    "cctv_frames_processed_total", "Frames processed", ["camera_id"]
)
frame_drops_total = Counter(
    "cctv_frame_drops_total", "Frames dropped because reader was behind", ["camera_id"]
)
inference_latency = Histogram(
    "cctv_inference_latency_seconds",
    "Wall time per detection+tracking+OCR pass",
    ["camera_id"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0),
)
ocr_attempts_total = Counter("cctv_ocr_attempts_total", "OCR attempts", ["camera_id"])
ocr_success_total = Counter("cctv_ocr_success_total", "OCR reads above auto-accept threshold", ["camera_id"])
violations_total = Counter(
    "cctv_violations_total", "Violations fired", ["camera_id", "code"]
)
camera_up = Gauge("cctv_camera_up", "1 if camera is producing frames, else 0", ["camera_id"])
last_frame_unixtime = Gauge(
    "cctv_camera_last_frame_unixtime", "Unix time of last frame from camera", ["camera_id"]
)

# ---- api-side
review_backlog = Gauge("cctv_review_backlog", "Pending violations awaiting review")
review_action_total = Counter("cctv_review_action_total", "Reviews submitted", ["action"])
api_request_latency = Histogram(
    "cctv_api_request_latency_seconds",
    "API request latency",
    ["method", "path", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)


_started = False
_lock = threading.Lock()


def start_metrics_server(port: int) -> None:
    """Idempotent — first call binds the port, later calls no-op."""
    global _started
    with _lock:
        if _started:
            return
        start_http_server(port)
        _started = True


class Timer:
    def __init__(self, hist: Histogram, *labels: str):
        self._hist = hist.labels(*labels) if labels else hist
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._hist.observe(time.perf_counter() - self._t0)
