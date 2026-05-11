"""Per-frame zero-shot violence classifier using OpenAI CLIP.

Why CLIP and not a fine-tuned binary classifier:
  - The previous attempt (jaranohaal/vit-base-violence-detection) shipped
    with timm-style key naming AND timm meta-tensor initialisation. Even
    after fixing both, fine-tuned-on-RLVS classifiers tend to label every
    crowded scene as violent because RLVS training negatives are mostly
    empty / quiet rooms. They don't see "busy street with cars" during
    training and over-fire on traffic footage.

  - CLIP was trained on 400 M image-text pairs covering everything from
    fights to traffic to empty fields. We use *competing prompts*:
        violent:  "two people fighting", "physical violence" ...
        non-violent: "cars driving on a road", "a normal street scene" ...
    A frame of cars on a road wins the "cars driving on a road" prompt,
    so violence probability stays low. A real fight wins violence
    prompts. Cars-as-violence false positives disappear *by construction*.

  - openai/clip-vit-base-patch32 loads cleanly via HuggingFace
    transformers — no meta-tensor warnings, no timm-vs-HF naming
    mismatch. ~600 MB download on first run, ~10 ms inference per frame
    on CUDA, ~50 ms on CPU.

References:
  - CLIP (Radford et al., 2021): https://arxiv.org/abs/2103.00020
  - Open-VCLIP (Weng et al., ICML 2023): https://proceedings.mlr.press/v202/weng23b/weng23b.pdf
  - ST-CLIP zero-shot action detection (Liang et al., WACV 2025):
    https://github.com/webber2933/ST-CLIP
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("violence.frame")

_CLIP_AVAILABLE = False
try:
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_AVAILABLE = True
except ImportError:
    pass

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Carefully balanced prompt set. Each side has roughly equal number of
# prompts so the softmax isn't biased. Negative prompts include the
# specific scenes our system has confused for violence in the past
# (cars, crowds, empty spaces) so they win on those frames.
VIOLENT_PROMPTS = [
    "a photo of two people fighting",
    "a photo of physical violence between people",
    "a photo of a person attacking another person",
    "a photo of people brawling in the street",
    "a photo of a person being assaulted",
    "a photo of a fight breaking out",
]

NORMAL_PROMPTS = [
    "a photo of cars driving on a road",
    "a photo of a normal street scene",
    "a photo of people walking calmly",
    "a photo of an empty road",
    "a photo of people standing quietly",
    "a photo of a peaceful crowd",
    "a photo of traffic on a highway",
    "a photo of an ordinary outdoor scene",
]


@dataclass
class FrameVerdict:
    score: float           # P(violent) over the prompt vocabulary
    is_violent: bool
    label: str             # top-matching prompt
    normal_score: float    # P(normal), useful for the "very-strong-single" rule


class FrameViolenceClassifier:
    """CLIP zero-shot violence classifier with sliding-window vote.

    Aggregation:
      - Maintain the last `window` per-frame verdicts per camera.
      - Camera is "confirmed violent" if ≥ `min_positive` of those
        verdicts crossed `per_frame_threshold`.

    Strong-single signal:
      - `is_strong_violent(camera_id)` returns True when the LATEST
        verdict has violent_score > strong_threshold AND normal_score
        is low enough that it's not a tied scene.
      - The orchestrator uses this to override the 2-of-3 rule.
    """

    def __init__(
        self,
        device: str = "cpu",
        per_frame_threshold: float = 0.55,
        strong_threshold: float = 0.75,
        strong_normal_max: float = 0.25,
        window: int = 8,
        min_positive: int = 4,
        run_every_n_frames: int = 6,
    ):
        self._device_str = device
        self._per_frame_threshold = per_frame_threshold
        self._strong_threshold = strong_threshold
        self._strong_normal_max = strong_normal_max
        self._window = window
        self._min_positive = min_positive
        self._run_every_n_frames = run_every_n_frames

        self._model = None
        self._processor = None
        self._text_features = None
        self._violent_count = len(VIOLENT_PROMPTS)
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=4)

        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._window))
        self._last_run_at: dict[str, int] = {}

        self._available = _CLIP_AVAILABLE
        if self._available:
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True, name="violence-frame-worker"
            )
            self._worker.start()
        if not self._available:
            log.warning("transformers not installed — CLIP frame classifier disabled")

    @property
    def available(self) -> bool:
        return self._available

    # ----------------------------------------------------------------

    def submit(self, camera_id: str, frame: np.ndarray, frame_idx: int) -> Optional[FrameVerdict]:
        if not self._available or frame is None or frame.size == 0:
            return None
        last = self._last_run_at.get(camera_id, -10_000)
        if (frame_idx - last) < self._run_every_n_frames:
            return None
        self._last_run_at[camera_id] = frame_idx
        try:
            queued = frame
            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 640:
                import cv2
                scale = 640.0 / max_dim
                queued = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            self._queue.put_nowait((camera_id, queued.copy(), frame_idx))
        except queue.Full:
            pass
        return None

    def is_confirmed(self, camera_id: str) -> bool:
        with self._history_lock:
            hist = list(self._history.get(camera_id, []))
        if not hist:
            return False
        return sum(1 for v in hist if v.is_violent) >= self._min_positive

    def is_strong_violent(self, camera_id: str) -> bool:
        """Latest verdict is high violence AND low normal — used to bypass
        the 2-of-3 fusion rule when CLIP is highly confident on its own."""
        with self._history_lock:
            hist = list(self._history.get(camera_id, []))
        if not hist:
            return False
        v = hist[-1]
        return v.score >= self._strong_threshold and v.normal_score <= self._strong_normal_max

    def latest_score(self, camera_id: str) -> float:
        with self._history_lock:
            hist = list(self._history.get(camera_id, []))
        return hist[-1].score if hist else 0.0

    def latest_label(self, camera_id: str) -> str:
        with self._history_lock:
            hist = list(self._history.get(camera_id, []))
        return hist[-1].label if hist else ""

    def reset(self, camera_id: str) -> None:
        with self._history_lock:
            self._history.pop(camera_id, None)
        self._last_run_at.pop(camera_id, None)

    def _worker_loop(self) -> None:
        while True:
            try:
                camera_id, frame, _frame_idx = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                verdict = self._infer(frame)
            except Exception:
                log.debug("CLIP frame inference failed", exc_info=True)
                continue
            with self._history_lock:
                self._history[camera_id].append(verdict)

    # ----------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                log.info("Loading CLIP zero-shot violence classifier (%s) — first run downloads ~600 MB",
                         CLIP_MODEL_ID)
                # use_safetensors=True forces HF to download model.safetensors
                # rather than falling back to pytorch_model.bin via torch.load
                # (which CVE-2025-32434 disables on torch < 2.6).
                model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True)
                processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
                device = torch.device(
                    self._device_str
                    if (self._device_str == "cpu" or torch.cuda.is_available())
                    else "cpu"
                )
                model = model.eval().to(device)
                self._model = model
                self._processor = processor
                self._device = device

                # Pre-compute text features once — they don't change between frames
                all_prompts = VIOLENT_PROMPTS + NORMAL_PROMPTS
                with torch.no_grad():
                    text_inputs = processor(
                        text=all_prompts, return_tensors="pt", padding=True
                    ).to(device)
                    text_features = model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self._text_features = text_features
                self._all_prompts = all_prompts
                log.info(
                    "CLIP ready on %s | %d violent prompts vs %d normal prompts | per-frame=%.2f strong=%.2f",
                    device, len(VIOLENT_PROMPTS), len(NORMAL_PROMPTS),
                    self._per_frame_threshold, self._strong_threshold,
                )
            except Exception:
                log.exception("CLIP frame classifier load failed — disabling")
                self._available = False

    def _infer(self, frame_bgr: np.ndarray) -> FrameVerdict:
        self._ensure_loaded()
        if not self._available or self._model is None:
            return FrameVerdict(score=0.0, is_violent=False, label="unavailable", normal_score=0.0)

        import cv2, torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            image_inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            image_features = self._model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity * CLIP logit scale (~100), then softmax over prompts
            logit_scale = self._model.logit_scale.exp()
            logits = (image_features @ self._text_features.T) * logit_scale
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

        violent_total = float(probs[:self._violent_count].sum())
        normal_total = float(probs[self._violent_count:].sum())
        top_idx = int(np.argmax(probs))
        top_prompt = self._all_prompts[top_idx]

        return FrameVerdict(
            score=violent_total,
            normal_score=normal_total,
            is_violent=violent_total >= self._per_frame_threshold,
            label=top_prompt,
        )
