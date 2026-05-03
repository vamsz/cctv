"""Download/prepare model weights into ./models.

Day-1 setup pulls only the COCO-trained YOLO11n. The helmet and plate
heads must be trained on labeled Indian-traffic data — this script
prints what to do for those.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    target = MODELS_DIR / "yolo11n.pt"
    if not target.exists():
        print(f"Downloading yolo11n.pt to {target} ...")
        # YOLO() downloads to CWD on first call; we move it.
        m = YOLO("yolo11n.pt")
        src = Path(m.ckpt_path) if hasattr(m, "ckpt_path") and m.ckpt_path else Path("yolo11n.pt")
        if src.resolve() != target.resolve():
            src.replace(target)
        print(f"OK: {target}")
    else:
        print(f"Already present: {target}")

    print()
    print("Helmet and plate heads are not auto-downloadable.")
    print("Train them on labeled data and place the .pt files at:")
    print(f"  {MODELS_DIR / 'helmet.pt'}")
    print(f"  {MODELS_DIR / 'plate.pt'}")
    print("Until then, the pipeline runs without them — helmet and plate-")
    print("based rules will simply not fire.")


if __name__ == "__main__":
    main()
