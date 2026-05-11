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

# Keywords for the UCF-Crime model's violence classes
_UCF_VIOLENCE_KEYWORDS = {"abuse", "assault", "fight", "robbery", "shooting",
                          "burglary", "vandalism", "stealing", "arson"}
# Labels that explicitly mean "no incident" — only these block confirmation
_UCF_NORMAL_KEYWORDS = {"normal"}

# Keywords for R3D-18 Kinetics-400 fallback
_K400_VIOLENCE_KEYWORDS = {
    "fight", "punch", "wrestl", "kick", "slap", "headbutt",
    "arm wrestling", "boxing", "karate", "judo", "sword", "shoot gun",
    "shove", "push", "throw", "hitting", "beating",
}
_K400_NORMAL_KEYWORDS: set[str] = set()    # Kinetics has no explicit "normal" class


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
        threshold: float = 0.35,
        model_name: str = "videomae",
    ):
        self._device_str = device
        self._threshold = threshold
        self._model = None
        self._backend = None    # "videomae" or "r3d18"
        self._violence_indices: list[int] = []
        self._normal_indices: list[int] = []
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
        queued = frame
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 384:
            import cv2
            scale = 384.0 / max_dim
            queued = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        with self._buf_lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque(maxlen=self.BUFFER_LEN)
            self._buffers[camera_id].append(queued.copy())

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
                self._normal_indices = [
                    i for i, label in self._id2label.items()
                    if any(kw in label.lower() for kw in _UCF_NORMAL_KEYWORDS)
                ]
                log.info(
                    "VideoMAE UCF-Crime loaded on %s | violence: %s | normal: %s | threshold=%.2f",
                    device,
                    [self._id2label[i] for i in self._violence_indices],
                    [self._id2label[i] for i in self._normal_indices],
                    self._threshold,
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
                self._normal_indices = []   # K400 has no explicit "normal" sentinel
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

                # ── Smart violence scoring (replaces naive sum-vs-threshold) ──
                # Three signals, OR'd together:
                #   (1) sum of all violence-class probabilities ≥ threshold
                #       AND the model's top-1 is NOT explicitly "Normal"
                #   (2) the single best violence class is itself the top-1
                #       (model is positively naming a violent action)
                #   (3) top violence class probability ≥ 1.5× threshold
                #       (handles cases where one class dominates)
                v_idx = self._violence_indices
                if v_idx:
                    v_probs = probs[v_idx]
                    violence_sum = float(v_probs.sum())
                    top_violence_local = int(np.argmax(v_probs))
                    top_violence_idx = v_idx[top_violence_local]
                    top_violence_prob = float(v_probs[top_violence_local])
                else:
                    violence_sum = 0.0
                    top_violence_idx = -1
                    top_violence_prob = 0.0

                top_idx = int(np.argmax(probs))
                top_label = self._id2label.get(top_idx, str(top_idx))
                top_is_normal = top_idx in self._normal_indices

                confirmed = (
                    (violence_sum >= self._threshold and not top_is_normal)
                    or (top_idx in v_idx and probs[top_idx] >= self._threshold * 0.6)
                    or (top_violence_prob >= self._threshold * 1.5)
                )

                # Score reported to the rest of the pipeline: max(sum, top_violence)
                # so even single-class spikes get reflected in the incident score.
                score = max(violence_sum, top_violence_prob)
                # Display label: the violence class with the highest probability
                # if the top-1 isn't already a violence class.
                if top_idx in v_idx:
                    label = top_label
                elif top_violence_idx >= 0:
                    label = (
                        f"{self._id2label.get(top_violence_idx, '?')} "
                        f"(top1={top_label})"
                    )
                else:
                    label = top_label

                log.debug(
                    "violence inference: backend=%s top=%s sum=%.3f best_v=%.3f confirmed=%s",
                    self._backend, top_label, violence_sum, top_violence_prob, confirmed,
                )

                result = ClipResult(
                    score=round(score, 3),
                    confirmed=confirmed,
                    label=label,
                    camera_id=camera_id,
                )
                try:
                    callback(result)
                except Exception:
                    log.debug("clip callback error", exc_info=True)
            except Exception:
                log.debug("clip inference error", exc_info=True)
