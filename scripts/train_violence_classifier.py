"""Fine-tune X3D-M on data/violence_ft/ for binary violence classification.

Architecture: torchvision X3D-M, pretrained on Kinetics-400. ~3.8 M
parameters, runs at ~25 FPS on RTX 3070 Ti. We freeze the backbone
and only train the new 2-class head + the last C2D block — keeps
the small-dataset (~175 train clips) from overfitting and reuses
Kinetics motion features that already encode "people doing things".

Strategy:
  - 16 frames sampled uniformly per video → temporal augmentation by
    random window start within each segment
  - Heavy spatial aug (color jitter + horizontal flip + random crop)
  - Mixup α=0.2 between Fight and NonFight clips
  - AdamW with cosine LR, 1e-4 base / 1e-3 head
  - ~50 epochs, ~30 min on RTX 3070 Ti
  - Save to models/violence_ft.pt

Usage:
  python scripts/train_violence_classifier.py
  # → models/violence_ft.pt + runs/violence_ft/

After training, set settings.violence_use_fine_tuned=True (auto-detected)
and the runner uses the fine-tuned classifier instead of CLIP zero-shot.

Reference dataset for augmentation (optional but big accuracy jump):
  - RWF-2000 (https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)
    2000 real CCTV clips, 80/20 train/test, MIT license
  - RLVS (https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
    2000 clips of real-life violence from YouTube
Place these under data/violence_ft/extra_train/Fight and ../NonFight to use.
"""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Performance env vars (must be set before torch import)
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 8))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models.video as tv_video

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_violence")

DATA_ROOT = ROOT / "data" / "violence_ft"
OUTPUT_PATH = ROOT / "models" / "violence_ft.pt"
RUNS_DIR = ROOT / "runs" / "violence_ft"

CLIP_LEN = 16
SIZE = 224
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def list_clips(split: str) -> list[tuple[Path, int]]:
    """Return [(video_path, label), ...] for `train` or `val`."""
    out: list[tuple[Path, int]] = []
    for lbl, folder in [(1, "Fight"), (0, "NonFight")]:
        d = DATA_ROOT / split / folder
        if not d.exists():
            continue
        for v in sorted(d.glob("*.mp4")):
            out.append((v, lbl))
    return out


def read_clip(
    path: Path,
    clip_len: int = CLIP_LEN,
    size: int = SIZE,
    augment: bool = False,
) -> Optional[np.ndarray]:
    """Sample `clip_len` frames uniformly, resize, return (T, H, W, C) uint8."""
    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        return None
    # Uniform sampling with optional temporal jitter
    if augment and n_frames > clip_len * 2:
        max_start = n_frames - clip_len
        start = random.randint(0, max_start // 2)
        end = n_frames - random.randint(0, max_start // 2)
    else:
        start, end = 0, n_frames - 1
    idxs = np.linspace(start, end, clip_len).astype(int)
    idx_set = set(idxs.tolist())
    frames: list[np.ndarray] = []
    cur = 0
    while cap.isOpened() and len(frames) < clip_len:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in idx_set:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cur += 1
    cap.release()
    while len(frames) < clip_len and frames:
        frames.append(frames[-1])         # repeat last if undershoot
    if not frames:
        return None
    return np.stack(frames, axis=0)


def augment_clip(clip: np.ndarray) -> np.ndarray:
    """Light spatial augmentation. clip is (T, H, W, C) uint8."""
    # Horizontal flip
    if random.random() < 0.5:
        clip = clip[:, :, ::-1, :]
    # Color jitter (whole-clip)
    if random.random() < 0.6:
        hsv = cv2.cvtColor(clip.reshape(-1, clip.shape[2], clip.shape[3]),
                           cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
        hsv[..., 1] *= random.uniform(0.8, 1.2)
        hsv[..., 2] *= random.uniform(0.8, 1.2)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        clip = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(clip.shape)
    return clip


def to_tensor(clip: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) uint8 → (C, T, H, W) float32 normalised."""
    x = clip.astype(np.float32) / 255.0
    x = (x - KINETICS_MEAN) / KINETICS_STD
    x = np.transpose(x, (3, 0, 1, 2))     # (C, T, H, W)
    return torch.from_numpy(np.ascontiguousarray(x))


class ViolenceDataset(Dataset):
    def __init__(self, split: str, augment: bool):
        self.clips = list_clips(split)
        self.augment = augment
        if not self.clips:
            raise RuntimeError(
                f"No clips found for split={split} under {DATA_ROOT}. "
                "Expected data/violence_ft/{split}/{Fight,NonFight}/*.mp4"
            )
        log.info("  %s: %d clips (Fight=%d, NonFight=%d)",
                 split, len(self.clips),
                 sum(1 for _, l in self.clips if l == 1),
                 sum(1 for _, l in self.clips if l == 0))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        path, label = self.clips[i]
        clip = read_clip(path, augment=self.augment)
        if clip is None:
            # try the next one — guards against the occasional unreadable file
            return self.__getitem__((i + 1) % len(self.clips))
        if self.augment:
            clip = augment_clip(clip)
        return to_tensor(clip), int(label)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Batch-wise mixup. Returns mixed_x, y_a, y_b, lam."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[perm]
    return mixed, y, y[perm], lam


def build_model() -> nn.Module:
    weights = tv_video.X3D_M_Weights.KINETICS400_V1 if hasattr(tv_video, 'X3D_M_Weights') else None
    try:
        model = tv_video.x3d_m(weights=weights) if weights else tv_video.x3d_m(pretrained=True)
    except Exception:
        # Older torchvision lacks X3D — fall back to R(2+1)D-18
        log.warning("X3D not available in this torchvision; falling back to R(2+1)D-18")
        w = tv_video.R2Plus1D_18_Weights.KINETICS400_V1
        model = tv_video.r2plus1d_18(weights=w)

    # Replace the 400-class Kinetics head with a 2-class violence head.
    # We expose this via a single Linear so the runtime model can be
    # softmaxed to get P(Fight).
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        # X3D from torchvision keeps the head under model.blocks[-1].proj.
        # Replace the projection's last Linear.
        head = model.blocks[-1]
        last_linear = None
        for m in head.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            in_features = last_linear.in_features
            # Walk parent module to swap
            for name, child in head.named_modules():
                if child is last_linear:
                    parent_name, _, attr = name.rpartition(".")
                    parent = head
                    for p in parent_name.split("."):
                        if p:
                            parent = getattr(parent, p)
                    setattr(parent, attr, nn.Linear(in_features, 2))
                    break
    if hasattr(model, "fc"):
        # R(2+1)D fallback
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the last block + the new head."""
    if hasattr(model, "blocks"):
        for i, block in enumerate(model.blocks):
            if i < len(model.blocks) - 2:
                for p in block.parameters():
                    p.requires_grad = False
    if hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad = False


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    log.info("listing dataset clips ...")

    train_ds = ViolenceDataset("train", augment=True)
    val_ds = ViolenceDataset("val", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    log.info("building X3D-M ...")
    model = build_model().to(device)
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("params: %d trainable / %d total (%.1f%%)",
             trainable, total, 100 * trainable / total)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    EPOCHS = 50

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y_a, y_b, lam = mixup(x, y, alpha=0.2)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            train_correct += (preds == y_a).sum().item()
            train_total += y_a.size(0)
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        log.info(
            "epoch %2d/%d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f  (%.0fs)",
            epoch + 1, EPOCHS, train_loss / max(train_total, 1),
            train_acc, val_acc, time.time() - t0,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "arch": "x3d_m",
                "labels": ["NonFight", "Fight"],
            }, OUTPUT_PATH)
            log.info("  ↳ new best (%.3f) saved to %s", val_acc, OUTPUT_PATH)

    log.info("done. best val_acc=%.3f. weights at %s", best_val_acc, OUTPUT_PATH)
    log.info(
        "To use the fine-tuned classifier, restart the pipeline; "
        "clip_classifier.py auto-loads models/violence_ft.pt when present."
    )


if __name__ == "__main__":
    main()
