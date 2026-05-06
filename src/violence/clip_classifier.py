"""Stage-2 violence clip classifier — async background worker.

Uses torchvision R3D-18 (3D ResNet-18, Apache 2.0, Kinetics-400 pretrained)
to confirm or reject a "suspected" violence incident via 16-frame clip inference.

Architecture per improvement plan:
  - Stage-1 (pose heuristics) raises "suspected" when ≥ 2 of 4 signals active.
  - Stage-2 (this module) samples last 2 s / 16 frames from a rolling buffer.
  - R3D-18 → Kinetics-400 class probabilities.
  - violence_score = sum(probs for violence-related K400 classes).
  - If score > threshold (default 0.55) → incident confirmed "active".
  - Runs in a background thread — does NOT block the main pipeline.

To fine-tune on RWF-2000 for best results (~89% accuracy per IDG-ViolenceNet 2025):
  python scripts/finetune_violence.py  --dataset data/rwf2000 --epochs 20

Apache-2.0 alternative: torchvision.models.video (R3D-18, MC3-18, S3D, R2Plus1D-18).
VideoMAE-base (CC-BY-NC) is NOT used to keep procurement-safe licensing.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("violence.clip")

_TORCH_VIDEO_AVAILABLE = False
try:
    import torch
    import torchvision.models.video as _tv_video
    _TORCH_VIDEO_AVAILABLE = True
except ImportError:
    pass

# ── Kinetics-400 violence-related class names (R3D-18 / MC3-18 class index → label)
# Loaded lazily from torchvision weights meta.
_VIOLENCE_KEYWORDS = {
    "fight", "punch", "wrestl", "kick", "slap", "headbutt",
    "arm wrestling", "boxing", "karate", "judo", "sword", "shoot gun",
    "shove", "push", "throw", "hitting", "beating",
}


def _violence_class_indices(weights) -> list[int]:
    """Return Kinetics-400 class indices whose names contain violence keywords."""
    cats = weights.meta.get("categories", [])
    idx = []
    for i, name in enumerate(cats):
        nl = name.lower()
        if any(kw in nl for kw in _VIOLENCE_KEYWORDS):
            idx.append(i)
    return idx or list(range(10))   # fallback: first 10 classes (safety net)


# ---------------------------------------------------------------------------

@dataclass
class ClipResult:
    score: float       # 0-1 violence probability
    confirmed: bool    # True if score > threshold → stage-1 should escalate
    label: str         # top matching Kinetics class name
    camera_id: str


class ViolenceClipClassifier:
    """Rolling-buffer async clip classifier.

    Call `submit_frame(camera_id, frame)` every pipeline frame.
    When stage-1 raises suspected, call `request_inference(camera_id, callback)`.
    The callback receives a `ClipResult` when inference completes.
    """

    BUFFER_LEN = 64     # frames kept in ring buffer (~4.3 s at 15 fps)
    CLIP_LEN = 16       # frames sampled per inference
    CLIP_SIZE = 112     # spatial resize (R3D-18 expects 112×112)

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.55,
        model_name: str = "r3d_18",
    ):
        self._device_str = device
        self._threshold = threshold
        self._model_name = model_name
        self._model = None
        self._violence_indices: list[int] = []
        self._weights = None

        # Per-camera rolling frame buffers
        self._buffers: dict[str, deque] = {}
        self._buf_lock = threading.Lock()

        # Inference queue: (camera_id, frames_snapshot, callback)
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._available = _TORCH_VIDEO_AVAILABLE

        if self._available:
            self._worker = threading.Thread(
                target=self._inference_loop, daemon=True, name="violence-clip-worker"
            )
            self._worker.start()
        else:
            log.warning(
                "torchvision not available — clip classifier disabled. "
                "Violence falls back to pose heuristics only."
            )

    # ---------------------------------------------------------------- public

    def submit_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Add one frame to this camera's rolling buffer (called every pipeline frame)."""
        if not self._available:
            return
        with self._buf_lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque(maxlen=self.BUFFER_LEN)
            self._buffers[camera_id].append(frame.copy())

    def request_inference(
        self,
        camera_id: str,
        callback: Callable[[ClipResult], None],
    ) -> None:
        """Enqueue a clip inference for camera_id. Non-blocking; result via callback."""
        if not self._available:
            return
        with self._buf_lock:
            buf = self._buffers.get(camera_id)
            frames_snapshot = list(buf) if buf else []

        if len(frames_snapshot) < 8:
            return   # not enough frames yet

        try:
            self._queue.put_nowait((camera_id, frames_snapshot, callback))
        except queue.Full:
            pass   # drop if worker is saturated

    @property
    def available(self) -> bool:
        return self._available

    # ---------------------------------------------------------------- private

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        device = torch.device(
            self._device_str
            if (self._device_str == "cpu" or torch.cuda.is_available())
            else "cpu"
        )
        try:
            weights_enum = {
                "r3d_18": _tv_video.R3D_18_Weights.KINETICS400_V1,
                "mc3_18": _tv_video.MC3_18_Weights.KINETICS400_V1,
                "r2plus1d_18": _tv_video.R2Plus1D_18_Weights.KINETICS400_V1,
            }.get(self._model_name, _tv_video.R3D_18_Weights.KINETICS400_V1)

            self._weights = weights_enum
            model_fn = {
                "r3d_18": _tv_video.r3d_18,
                "mc3_18": _tv_video.mc3_18,
                "r2plus1d_18": _tv_video.r2plus1d_18,
            }.get(self._model_name, _tv_video.r3d_18)

            self._model = model_fn(weights=weights_enum)
            self._model.eval().to(device)
            self._device = device
            self._violence_indices = _violence_class_indices(weights_enum)
            log.info(
                "Clip classifier: %s on %s | violence classes: %d",
                self._model_name, device, len(self._violence_indices),
            )
        except Exception:
            log.exception("Video model load failed — clip classifier disabled")
            self._available = False

    def _sample_clip(self, frames: list[np.ndarray]) -> "torch.Tensor":
        """Sample CLIP_LEN evenly-spaced frames, resize, and convert to tensor."""
        import torch
        n = len(frames)
        indices = [int(i * (n - 1) / (self.CLIP_LEN - 1)) for i in range(self.CLIP_LEN)]
        import cv2
        clip = []
        for idx in indices:
            f = cv2.resize(frames[idx], (self.CLIP_SIZE, self.CLIP_SIZE))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            clip.append(f)

        # (T, H, W, C) → (C, T, H, W) float32 normalised
        arr = np.stack(clip, axis=0).astype(np.float32) / 255.0
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1,C,T,H,W)
        return tensor.to(self._device)

    def _inference_loop(self) -> None:
        self._load_model()
        while True:
            try:
                camera_id, frames_snapshot, callback = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self._available or self._model is None:
                continue
            try:
                import torch
                clip = self._sample_clip(frames_snapshot)
                with torch.no_grad():
                    logits = self._model(clip)[0]
                    probs = torch.softmax(logits, dim=0).cpu().numpy()

                if self._violence_indices:
                    score = float(np.sum(probs[self._violence_indices]))
                else:
                    score = 0.0

                top_idx = int(np.argmax(probs))
                cats = self._weights.meta.get("categories", [])
                top_label = cats[top_idx] if top_idx < len(cats) else str(top_idx)

                result = ClipResult(
                    score=round(score, 3),
                    confirmed=score >= self._threshold,
                    label=top_label,
                    camera_id=camera_id,
                )
                try:
                    callback(result)
                except Exception:
                    log.debug("clip callback error", exc_info=True)
            except Exception:
                log.debug("clip inference error", exc_info=True)
