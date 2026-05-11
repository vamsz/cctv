"""RTSP / file / webcam ingest with a background reader thread.

OpenCV's VideoCapture blocks on reads. For RTSP, that means if we read
synchronously inside the inference loop and inference takes longer than
the camera's frame interval, we accumulate a backlog of stale frames in
the OS socket buffer and end up showing 5-second-old footage. The fix is
to run the capture on its own thread that always discards in favor of
the latest frame. The inference loop reads "the most recent frame" — if
there is no new frame, it skips, never blocks.

Looping: for file sources, when EOF is reached we rewind to frame 0
and keep streaming. This is useful for testing with recorded videos.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("ingest.stream")


@dataclass
class StreamConfig:
    camera_id: str
    source: str | int
    fps_cap: int = 30
    reconnect_delay_seconds: float = 2.0
    loop: bool = False  # If True, rewind to start on EOF (for video files)


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

        # Inference reads: track last frame index consumed by inference
        self._inference_last_idx: int = 0

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
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except Exception:
            pass
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
                if self.cfg.loop:
                    # Rewind to beginning for looping video files
                    try:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        log.debug("[%s] video EOF — looping", self.cfg.camera_id)
                    except Exception:
                        self._cap.release()
                        self._cap = None
                    continue
                else:
                    # Stream died — for files this means EOF, for RTSP it usually
                    # means the camera dropped. Reopen and keep going.
                    self._cap.release()
                    self._cap = None
                    time.sleep(self.cfg.reconnect_delay_seconds)
                    continue

            now = time.time()
            if now - last_emit < self._fps_interval:
                time.sleep(min(self._fps_interval - (now - last_emit), 0.005))
                continue
            last_emit = now

            with self._lock:
                self._frame = frame
                self._frame_idx += 1
                self._timestamp = now

    def read(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """Return (frame_index, unix_timestamp, frame_bgr) or None if no frame yet.
        
        This returns the latest frame every time — used for MJPEG streaming.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame_idx, self._timestamp, self._frame.copy()

    def read_for_inference(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """Return the latest frame only if it hasn't been consumed by inference yet.
        
        Returns None if inference has already seen this frame index.
        This prevents inference from re-processing the same frame and 
        allows it to naturally skip frames it can't keep up with — 
        without logging "dropped" warnings since the stream itself is smooth.
        """
        with self._lock:
            if self._frame is None:
                return None
            if self._frame_idx <= self._inference_last_idx:
                return None
            self._inference_last_idx = self._frame_idx
            return self._frame_idx, self._timestamp, self._frame.copy()

    def __enter__(self) -> "StreamReader":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
