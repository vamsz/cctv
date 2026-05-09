"""Stage-2 violence clip classifier — async background worker.

Primary model: VideoMAE fine-tuned on UCF-Crime CCTV dataset
  HuggingFace: OPear/videomae-large-finetuned-UCF-Crime
  92.96% validation accuracy, 14 crime classes, trained on real surveillance footage.
  Classes include: Fighting, Assault, Abuse, Robbery, Shooting + Normal Videos, etc.

Fallback: torchvision R3D-18 Kinetics-400 (if transformers not installed).

Runs in a daemon background thread — does NOT block the main pipeline.

Install primary model:
    pip install transformers
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger("violence.clip")

_VIDEOMAE_AVAILABLE = False
try:
    from transformers import VideoMAEForVideoClassification as _VideoMAEModel
    _VIDEOMAE_AVAILABLE = True
except ImportError:
    pass

_TORCH_VIDEO_AVAILABLE = False
try:
    import torch
    import torchvision.models.video as _tv_video
    _TORCH_VIDEO_AVAILABLE = True
except ImportError:
    pass

VIDEOMAE_MODEL_ID = "OPear/videomae-large-finetuned-UCF-Crime"

# Keywords to match against UCF-Crime label names → violence score
_UCF_VIOLENCE_KEYWORDS = {"abuse", "assault", "fight", "robbery", "shooting"}

# Keywords for R3D-18 Kinetics-400 fallback
_K400_VIOLENCE_KEYWORDS = {
    "fight", "punch", "wrestl", "kick", "slap", "headbutt",
    "arm wrestling", "boxing", "karate", "judo", "sword", "shoot gun",
    "shove", "push", "throw", "hitting", "beating",
}


@dataclass
class ClipResult:
    score: float       # 0-1 violence probability
    confirmed: bool    # True if score > threshold
    label: str         # top predicted class name
    camera_id: str


class ViolenceClipClassifier:
    """Rolling-buffer async clip classifier.

    Call `submit_frame(camera_id, frame)` every pipeline frame.
    When stage-1 raises suspected, call `request_inference(camera_id, callback)`.
    The callback receives a `ClipResult` when inference completes.
    """

    BUFFER_LEN = 64     # frames kept in ring buffer (~4.3 s at 15 fps)
    CLIP_LEN = 16       # frames sampled per inference

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.55,
        model_name: str = "videomae",
    ):
        self._device_str = device
        self._threshold = threshold
        self._model = None
        self._backend = None    # "videomae" or "r3d18"
        self._violence_indices: list[int] = []
        self._id2label: dict[int, str] = {}

        self._buffers: dict[str, deque] = {}
        self._buf_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._available = _VIDEOMAE_AVAILABLE or _TORCH_VIDEO_AVAILABLE

        if self._available:
            self._worker = threading.Thread(
                target=self._inference_loop, daemon=True, name="violence-clip-worker"
            )
            self._worker.start()
        else:
            log.warning(
                "No video model backend available — clip classifier disabled. "
                "Install: pip install transformers torch torchvision"
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
            return

        try:
            self._queue.put_nowait((camera_id, frames_snapshot, callback))
        except queue.Full:
            pass

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
        self._device = device

        # Primary: VideoMAE fine-tuned on UCF-Crime CCTV dataset
        if _VIDEOMAE_AVAILABLE:
            try:
                from transformers import VideoMAEForVideoClassification
                log.info(
                    "Loading VideoMAE UCF-Crime model from HuggingFace (%s) — "
                    "first run downloads ~1.2 GB...", VIDEOMAE_MODEL_ID
                )
                model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL_ID)
                model = model.eval().to(device)
                self._model = model
                self._backend = "videomae"
                self._id2label = {int(k): v for k, v in model.config.id2label.items()}
                self._violence_indices = [
                    i for i, label in self._id2label.items()
                    if any(kw in label.lower() for kw in _UCF_VIOLENCE_KEYWORDS)
                ]
                log.info(
                    "VideoMAE UCF-Crime loaded on %s | violence classes: %s",
                    device,
                    [self._id2label[i] for i in self._violence_indices],
                )
                return
            except Exception:
                log.exception("VideoMAE load failed — falling back to R3D-18 Kinetics-400")

        # Fallback: R3D-18 Kinetics-400
        if _TORCH_VIDEO_AVAILABLE:
            try:
                weights = _tv_video.R3D_18_Weights.KINETICS400_V1
                model = _tv_video.r3d_18(weights=weights).eval().to(device)
                self._model = model
                self._backend = "r3d18"
                self._weights = weights
                cats = weights.meta.get("categories", [])
                self._id2label = {i: cats[i] for i in range(len(cats))}
                self._violence_indices = [
                    i for i, name in enumerate(cats)
                    if any(kw in name.lower() for kw in _K400_VIOLENCE_KEYWORDS)
                ] or list(range(10))
                log.info(
                    "R3D-18 Kinetics-400 loaded on %s (fallback) | violence classes: %d",
                    device, len(self._violence_indices),
                )
            except Exception:
                log.exception("R3D-18 load failed — classifier disabled")
                self._available = False

    def _sample_clip(self, frames: list[np.ndarray]) -> "torch.Tensor":
        """Sample CLIP_LEN evenly-spaced frames, resize, normalise, return tensor."""
        import torch, cv2

        n = len(frames)
        indices = [int(i * (n - 1) / (self.CLIP_LEN - 1)) for i in range(self.CLIP_LEN)]

        if self._backend == "videomae":
            size = 224
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            size = 112
            mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
            std  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        clip = []
        for idx in indices:
            f = cv2.resize(frames[idx], (size, size))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            f = (f - mean) / std
            clip.append(f)

        arr = np.stack(clip, axis=0)  # (T, H, W, C)

        if self._backend == "videomae":
            # VideoMAE expects (batch, num_frames, channels, H, W)
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
        else:
            # R3D-18 expects (batch, C, T, H, W)
            tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)

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
                    if self._backend == "videomae":
                        outputs = self._model(pixel_values=clip)
                        probs = torch.softmax(outputs.logits[0], dim=0).cpu().numpy()
                    else:
                        logits = self._model(clip)[0]
                        probs = torch.softmax(logits, dim=0).cpu().numpy()

                score = (
                    float(np.sum(probs[self._violence_indices]))
                    if self._violence_indices else 0.0
                )
                top_idx = int(np.argmax(probs))
                top_label = self._id2label.get(top_idx, str(top_idx))

                log.debug(
                    "violence inference: backend=%s top=%s score=%.3f confirmed=%s",
                    self._backend, top_label, score, score >= self._threshold,
                )

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
