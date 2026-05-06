"""Vehicle crop → L2-normalised embedding for cross-camera ReID.

Primary backbone: DINOv2-Small (facebook/dinov2-small, ViT-S/14, MIT, ~22M params).
  - CLS-token embedding: 384-d
  - ~5-10 ms/crop RTX 3060; ~20 ms CPU (batch=1)
  - Drop-in upgrade: cosine similarity threshold should be 0.70 (was 0.85 for MobileNet)

Fallback: MobileNetV3-Small (torchvision, IMAGENET1K_V1) when transformers is not installed.
  - 576-d embedding, ~1.5 ms GPU, ~18 ms CPU

Per improvement plan: DINOv2-Small improves cross-camera Rank-1 +10-20% on hard cases
vs MobileNetV3-Small (TransReID 81.7% mAP VeRi-776 baseline as reference).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("reid.embedder")

_DINO_AVAILABLE = False
try:
    from transformers import AutoModel, AutoImageProcessor  # pip install transformers
    _DINO_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# DINOv2-Small backbone
# ---------------------------------------------------------------------------

class _DINOv2Embedder:
    DIM = 384

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        model_id = "facebook/dinov2-small"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval().to(self.device)
        log.info("ReID embedder: DINOv2-Small (384-d, MIT) on %s", self.device)

    @torch.no_grad()
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]   # (1, 384)
        normed = F.normalize(cls_token, p=2, dim=1)
        return normed.cpu().numpy()[0]                    # (384,)


# ---------------------------------------------------------------------------
# MobileNetV3-Small fallback
# ---------------------------------------------------------------------------

class _MobileNetEmbedder:
    DIM = 576

    def __init__(self, device: str = "cpu"):
        import torchvision.models as tv
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        weights = tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = tv.mobilenet_v3_small(weights=weights)
        self.backbone = nn.Sequential(*list(model.features), model.avgpool)
        self.backbone.eval().to(self.device)
        self._tf = weights.transforms()
        log.info("ReID embedder: MobileNetV3-Small fallback (576-d) on %s", self.device)

    @torch.no_grad()
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._tf(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        feat = self.backbone(tensor).flatten(1)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Public interface (unchanged from caller perspective)
# ---------------------------------------------------------------------------

class VehicleEmbedder:
    """Wraps DINOv2-Small (preferred) or MobileNetV3-Small (fallback)."""

    def __init__(self, device: str = "cpu"):
        if _DINO_AVAILABLE:
            try:
                self._impl = _DINOv2Embedder(device)
                self.DIM = _DINOv2Embedder.DIM
                return
            except Exception:
                log.warning("DINOv2-Small init failed — using MobileNetV3-Small", exc_info=True)
        self._impl = _MobileNetEmbedder(device)
        self.DIM = _MobileNetEmbedder.DIM

    @property
    def backend(self) -> str:
        return "dinov2-small" if isinstance(self._impl, _DINOv2Embedder) else "mobilenetv3-small"

    def extract(
        self,
        frame: np.ndarray,
        xyxy: tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return None
        return self._run(crop)

    def extract_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return None
        return self._run(crop)

    def _run(self, crop_bgr: np.ndarray) -> np.ndarray:
        try:
            return self._impl.embed(crop_bgr)
        except Exception:
            log.debug("embedding failed", exc_info=True)
            return np.zeros(self.DIM, dtype=np.float32)
