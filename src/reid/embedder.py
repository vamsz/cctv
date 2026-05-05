"""Vehicle crop → L2-normalised embedding for cross-camera ReID.

Backbone: MobileNetV3-Small (torchvision, IMAGENET1K_V1 weights).
The classifier head is removed; the 576-dim pooled feature is
L2-normalised — so cosine similarity == dot product.

On RTX 3070Ti: ~1.5 ms per crop (GPU, batch=1).
On CPU: ~18 ms per crop. Still usable when extracted every N frames.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class VehicleEmbedder:
    DIM = 576   # MobileNetV3-Small last-conv output channels

    def __init__(self, device: str = "cpu"):
        import torchvision.models as tv
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        weights = tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = tv.mobilenet_v3_small(weights=weights)
        # Drop classifier; keep feature extractor + adaptive pool
        self.backbone = nn.Sequential(*list(model.features), model.avgpool)
        self.backbone.eval().to(self.device)
        self._tf = weights.transforms()   # resize + centercrop + normalize

    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        xyxy: tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Return 576-dim L2-normalised embedding, or None if crop too small."""
        x1, y1, x2, y2 = (max(0, int(v)) for v in xyxy)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return None
        return self._run(crop)

    @torch.no_grad()
    def extract_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return None
        return self._run(crop)

    def _run(self, crop_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._tf(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        feat = self.backbone(tensor).flatten(1)                   # (1, 576)
        feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0]                              # (576,)
