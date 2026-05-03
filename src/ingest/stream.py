"""RTSP / file / webcam ingest with a background reader thread.

OpenCV's VideoCapture blocks on reads. For RTSP, that means if we read
synchronously inside the inference loop and inference takes longer than
the camera's frame interval, we accumulate a backlog of stale frames in
the OS socket buffer and end up showing 5-second-old footage. The fix is
to run the capture on its own thread that always discards in favor of
the latest frame. The inference loop reads "the most recent frame" — if
there is no new frame, it skips, never blocks.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StreamConfig:
    camera_id: str
    source: str | int
    fps_cap: int = 15
    reconnect_delay_seconds: float = 2.0


class StreamReader:
    def __init__(self, cfg: StreamConfig):
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_idx: int = 0
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_interval = 1.0 / max(self.cfg.fps_cap, 1)

    def start(self) -> "StreamReader":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"reader-{self.cfg.camera_id}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def _open(self) -> bool:
        src = self.cfg.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return False
        # Keep buffer minimal so we always read the freshest frame.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _loop(self) -> None:
        last_emit = 0.0
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open():
                    time.sleep(self.cfg.reconnect_delay_seconds)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Stream died — for files this means EOF, for RTSP it usually
                # means the camera dropped. Reopen and keep going.
                self._cap.release()
                self._cap = None
                time.sleep(self.cfg.reconnect_delay_seconds)
                continue

            now = time.time()
            if now - last_emit < self._fps_interval:
                continue
            last_emit = now

            with self._lock:
                self._frame = frame
                self._frame_idx += 1
                self._timestamp = now

    def read(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """Return (frame_index, unix_timestamp, frame_bgr) or None if no frame yet."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame_idx, self._timestamp, self._frame.copy()

    def __enter__(self) -> "StreamReader":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
