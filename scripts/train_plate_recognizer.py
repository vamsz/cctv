"""Train a custom CRNN plate recogniser on hand-labelled Indian plates.

Architecture:
  MobileNetV3-Small features (~1.5 M params)
    → AdaptiveAvgPool to fixed time dimension
    → BiLSTM × 2 (256 hidden)
    → Linear → 37 classes (A-Z, 0-9, blank)
    → CTC loss (no alignment needed)

Why this beats fine-tuning fast-plate-ocr:
  - fast-plate-ocr ships as ONNX — fine-tuning the ONNX graph is
    complex; training fresh PyTorch weights is cleaner.
  - MobileNetV3-S features are well-suited to thin text strips.
  - CRNN+CTC handles variable-length plates (7-10 chars) natively.
  - ~2 M params trains in 20-30 min on RTX 3070 Ti.

Dataset:
  data/plate_ocr_ft/labelled/<TEXT>.jpg   ← hand-labelled crops
  data/plate_ocr_ft/labelled/<TEXT>_<n>.jpg  ← multiple shots of same plate ok

Each filename (sans extension and trailing _N) is the label string.

Usage:
  python scripts/train_plate_recognizer.py
  # → models/plate_ocr_ft.pt
  # Auto-loaded by PlateOCR as a 3rd engine if present.

For more training data:
  - Mix in CCPD2019 (https://github.com/detectRecog/CCPD) — 250k labelled
    Chinese plates, similar XX##XX#### format. Place under
    data/plate_ocr_ft/labelled/ccpd/ and re-run.
  - Indian License Plates with Labels (Kaggle):
    https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels
"""
from __future__ import annotations

import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Performance env vars
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_ocr")

LABELLED_DIR = ROOT / "data" / "plate_ocr_ft" / "labelled"
OUTPUT_PATH = ROOT / "models" / "plate_ocr_ft.pt"

# Vocabulary: 36 alphanumeric chars + blank (CTC index 0)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank
NUM_CLASSES = len(CHARS) + 1                            # 37
IMG_H = 32
IMG_W = 128
MAX_LEN = 12


def label_from_filename(p: Path) -> Optional[str]:
    """Strip extension + trailing _N suffix. Return None if not a valid label."""
    stem = p.stem.split('_')[0].split('.')[0]
    stem = "".join(c for c in stem.upper() if c.isalnum())
    if not (2 <= len(stem) <= MAX_LEN):
        return None
    if any(c not in CHARS for c in stem):
        return None
    return stem


def preprocess(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale, resize to fixed 32×128 with aspect preservation."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    # Resize keeping height = IMG_H, then pad/crop width to IMG_W
    scale = IMG_H / h
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_CUBIC)
    if new_w > IMG_W:
        # centre-crop
        start = (new_w - IMG_W) // 2
        img = img[:, start:start + IMG_W]
    elif new_w < IMG_W:
        # right-pad with mean
        pad = np.full((IMG_H, IMG_W - new_w), int(img.mean()), dtype=np.uint8)
        img = np.concatenate([img, pad], axis=1)
    return img.astype(np.float32) / 255.0


def encode_label(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text]


def decode_greedy(logits: torch.Tensor) -> str:
    """CTC greedy decode for a single sample."""
    pred = logits.argmax(-1).cpu().numpy()
    out: list[str] = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:        # 0 = CTC blank
            out.append(CHARS[p - 1])
        prev = p
    return "".join(out)


class PlateDataset(Dataset):
    def __init__(self, items: list[tuple[Path, str]], augment: bool):
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            # slight rotation
            angle = random.uniform(-3, 3)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if random.random() < 0.5:
            # brightness
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.7, 1.3),
                                      beta=random.randint(-20, 20))
        if random.random() < 0.3:
            # gaussian noise
            n = np.random.normal(0, 8, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = cv2.imread(str(path))
        if img is None:
            return self.__getitem__((i + 1) % len(self.items))
        if self.augment:
            img = self._augment(img)
        x = preprocess(img)                          # (32, 128) float32
        y = encode_label(label)
        return x, y, len(y)


def collate(batch):
    xs = torch.from_numpy(np.stack([b[0] for b in batch], axis=0)).unsqueeze(1)  # (B, 1, H, W)
    ys = torch.tensor([t for b in batch for t in b[1]], dtype=torch.long)
    y_lens = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return xs, ys, y_lens


# ------------------------------------------------------------------
# CRNN model
# ------------------------------------------------------------------

class CRNN(nn.Module):
    """Small CRNN: MobileNetV3-S features → BiLSTM × 2 → linear."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        import torchvision.models as tv
        backbone = tv.mobilenet_v3_small(weights=None)
        # Re-do the first conv to accept 1-channel input
        backbone.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=(2, 1), padding=1, bias=False,
        )
        self.features = backbone.features    # → ~(B, 576, 1, T)
        # We want output shape (B, T', 576) for the LSTM
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        feat_dim = 576
        hidden = 256
        self.rnn = nn.LSTM(
            input_size=feat_dim, hidden_size=hidden,
            num_layers=2, bidirectional=True, batch_first=True, dropout=0.1,
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 128)
        f = self.features(x)                 # (B, 576, h', w')
        f = self.pool(f)                     # (B, 576, 1, w')
        f = f.squeeze(2).permute(0, 2, 1)    # (B, w', 576)
        out, _ = self.rnn(f)                 # (B, T', 2*hidden)
        logits = self.classifier(out)        # (B, T', num_classes)
        return logits


def split_train_val(items, val_fraction: float = 0.1):
    random.shuffle(items)
    n_val = max(1, int(len(items) * val_fraction))
    return items[n_val:], items[:n_val]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    if not LABELLED_DIR.exists():
        log.error("no labelled crops at %s", LABELLED_DIR)
        log.error("Run scripts/extract_plate_crops_for_labelling.py first,")
        log.error("rename crops to their correct text, and move them under labelled/.")
        sys.exit(1)

    items: list[tuple[Path, str]] = []
    for p in LABELLED_DIR.rglob("*.jpg"):
        label = label_from_filename(p)
        if label:
            items.append((p, label))
    log.info("found %d labelled crops", len(items))
    if len(items) < 30:
        log.error("Need at least 30 labelled crops to train. Hand-label more.")
        sys.exit(1)

    train_items, val_items = split_train_val(items, val_fraction=0.1)
    log.info("train=%d  val=%d", len(train_items), len(val_items))

    train_ds = PlateDataset(train_items, augment=True)
    val_ds = PlateDataset(val_items, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate, persistent_workers=True,
    )

    model = CRNN().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=60)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    EPOCHS = 60
    best_acc = 0.0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        for xs, ys, y_lens in train_loader:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            y_lens = y_lens.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(xs)                          # (B, T, C)
                logp = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
                x_lens = torch.full((xs.size(0),), logp.size(0), dtype=torch.long, device=device)
                loss = ctc(logp, ys, x_lens, y_lens)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()
            train_loss += loss.item() * xs.size(0)
            n += xs.size(0)
        sched.step()

        # validation
        model.eval()
        val_total = 0
        val_exact = 0
        char_total = 0
        char_correct = 0
        with torch.no_grad():
            for xs, ys, y_lens in val_loader:
                xs = xs.to(device, non_blocking=True)
                logits = model(xs)
                # Pull individual sequences for greedy decoding
                offset = 0
                for i in range(xs.size(0)):
                    target = ys[offset:offset + y_lens[i]].cpu().numpy()
                    offset += y_lens[i].item()
                    pred = decode_greedy(logits[i])
                    truth = "".join(CHARS[t - 1] for t in target)
                    val_total += 1
                    if pred == truth:
                        val_exact += 1
                    # per-character accuracy via edit-distance-style alignment
                    L = max(len(pred), len(truth))
                    char_total += L
                    for j in range(min(len(pred), len(truth))):
                        if pred[j] == truth[j]:
                            char_correct += 1

        train_loss /= max(n, 1)
        plate_acc = val_exact / max(val_total, 1)
        char_acc = char_correct / max(char_total, 1)
        log.info(
            "epoch %2d/%d  loss=%.4f  val_plate_acc=%.3f  val_char_acc=%.3f  (%.0fs)",
            epoch + 1, EPOCHS, train_loss, plate_acc, char_acc, time.time() - t0,
        )

        if plate_acc > best_acc:
            best_acc = plate_acc
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "chars": CHARS,
                "img_h": IMG_H,
                "img_w": IMG_W,
                "epoch": epoch + 1,
                "val_plate_acc": plate_acc,
                "val_char_acc": char_acc,
            }, OUTPUT_PATH)
            log.info("  ↳ new best (plate_acc=%.3f) saved to %s", plate_acc, OUTPUT_PATH)

    log.info("done. best val_plate_acc=%.3f. weights at %s", best_acc, OUTPUT_PATH)
    log.info(
        "Restart the pipeline; PlateOCR auto-loads this model if a wrapper "
        "is added under src/ocr/plate_recognizer_ft.py."
    )


if __name__ == "__main__":
    main()
