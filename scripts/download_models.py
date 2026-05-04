"""Download all model weights into ./models.

Downloads:
  1. yolo11n.pt          — COCO general detector (vehicles, persons, etc.)
  2. plate.pt            — License plate detector (from HuggingFace)
  3. helmet.pt           — Helmet/no-helmet detector (from HuggingFace)

Usage:
    python scripts/download_models.py
"""
from pathlib import Path
import sys
import urllib.request
import shutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_MODELS = {
    "plate.pt": {
        "url": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1m.pt",
        "description": "YOLO11m license plate detector (morsetechlab, 40MB, mAP@50=0.98, best accuracy)",
    },
    "helmet.pt": {
        "url": "https://huggingface.co/nnsohamnn/helmet-detection-yolo11/resolve/main/yolov11s(80%20epochs).pt",
        "description": "YOLO11s helmet detector (nnsohamnn, 19MB, mAP@50=0.89, classes: With Helmet / Without Helmet)",
    },
}


def download_file(url: str, dest: Path, description: str) -> bool:
    if dest.exists():
        print(f"  Already present: {dest.name}")
        return True
    print(f"  Downloading {dest.name}...")
    print(f"    {description}")
    print(f"    From: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print(f"Model directory: {MODELS_DIR}\n")

    # 1. General YOLO11n (COCO)
    print("[1/3] General detector (COCO — vehicles, persons, etc.)")
    target = MODELS_DIR / "yolo11n.pt"
    if not target.exists():
        from ultralytics import YOLO
        m = YOLO("yolo11n.pt")
        src = Path(m.ckpt_path) if hasattr(m, "ckpt_path") and m.ckpt_path else Path("yolo11n.pt")
        if src.resolve() != target.resolve():
            src.replace(target)
        print(f"  OK: {target.name}")
    else:
        print(f"  Already present: {target.name}")

    # 2. License plate detector
    print("\n[2/3] License plate detector")
    plate_info = HUGGINGFACE_MODELS.get("plate.pt", {})
    if plate_info.get("url"):
        download_file(plate_info["url"], MODELS_DIR / "plate.pt", plate_info["description"])
    else:
        print("  No URL configured — skipping")

    # 3. Helmet detector
    print("\n[3/3] Helmet / no-helmet detector")
    helmet_info = HUGGINGFACE_MODELS["helmet.pt"]
    download_file(helmet_info["url"], MODELS_DIR / "helmet.pt", helmet_info["description"])

    # Summary
    print("\n" + "=" * 50)
    print("Model status:")
    for name in ["yolo11n.pt", "plate.pt", "helmet.pt"]:
        path = MODELS_DIR / name
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name:<16} ({size:.1f} MB)")
        else:
            print(f"  [MISSING] {name:<16}")
    print("=" * 50)


if __name__ == "__main__":
    main()
