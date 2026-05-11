"""Generate synthetic Indian license plates to supplement OCR training data.

This script creates realistic license plate crops by rendering text with
standard fonts, applying random affine transformations, and adding noise/blur
to simulate low-quality CCTV captures. 

Usage:
  python scripts/generate_synthetic_plates.py --count 5000
"""
import argparse
import logging
import os
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synthetic_plates")

OUT_DIR = ROOT / "data" / "plate_ocr_ft" / "labelled" / "synthetic"
STATES = [
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", 
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", 
    "TR", "UP", "UK", "WB", "DL", "CH"
]

def generate_plate_text() -> str:
    """Generate a valid Indian license plate text."""
    state = random.choice(STATES)
    district = f"{random.randint(1, 99):02d}"
    letters_len = random.choice([1, 2])
    letters = "".join(random.choices(string.ascii_uppercase, k=letters_len))
    numbers = f"{random.randint(1, 9999):04d}"
    return f"{state}{district}{letters}{numbers}"

def add_noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise and slight blur to simulate CCTV feed."""
    if random.random() < 0.7:
        noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    if random.random() < 0.5:
        ksize = random.choice([(3, 3), (5, 5)])
        img = cv2.GaussianBlur(img, ksize, random.uniform(0.5, 1.5))
        
    if random.random() < 0.3:
        # Motion blur
        kernel_size = random.choice([3, 5])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        img = cv2.filter2D(img, -1, kernel)
        
    return img

def apply_affine(img: np.ndarray) -> np.ndarray:
    """Apply slight rotation, scaling, and translation."""
    h, w = img.shape[:2]
    angle = random.uniform(-4, 4)
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-5, 5)
    ty = random.uniform(-2, 2)
    
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    args = ap.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to load Arial Bold, fallback to default if missing
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    try:
        font = ImageFont.truetype(font_path, 22)
    except IOError:
        log.warning("Arial Bold not found. Using default font.")
        font = ImageFont.load_default()

    log.info("Generating %d synthetic Indian plates...", args.count)
    
    for i in range(args.count):
        text = generate_plate_text()
        
        # Create blank white plate (128x32 is the CRNN target size)
        # We make it slightly larger then resize
        w, h = 136, 40
        
        # Add random background tint (mostly white/yellow/gray)
        bg_color = (
            random.randint(200, 255), 
            random.randint(200, 255), 
            random.randint(180, 255)
        )
        img_pil = Image.new('RGB', (w, h), color=bg_color)
        draw = ImageDraw.Draw(img_pil)
        
        # Draw text centered
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # If text is too long, we scale it
        x = max((w - text_w) / 2, 2)
        y = max((h - text_h) / 2 - 4, 0)
        
        # Random text color (dark gray to black)
        text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        draw.text((x, y), text, font=font, fill=text_color)
        
        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        img_cv = apply_affine(img_cv)
        img_cv = add_noise(img_cv)
        
        # Resize to final CRNN dimensions
        img_cv = cv2.resize(img_cv, (128, 32), interpolation=cv2.INTER_AREA)
        
        out_path = OUT_DIR / f"{text}_{i}.jpg"
        cv2.imwrite(str(out_path), img_cv)
        
        if (i + 1) % 500 == 0:
            log.info("Generated %d / %d plates...", i + 1, args.count)
            
    log.info("Successfully generated %d synthetic plates in %s", args.count, OUT_DIR)

if __name__ == "__main__":
    main()
