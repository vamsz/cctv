"""Fine-tune a YOLO model for helmet / no_helmet detection.

Expected dataset layout (Ultralytics YOLO format):

    datasets/helmet/
        images/{train,val}/...jpg
        labels/{train,val}/...txt   # YOLO txt: class cx cy w h (normalized)
        data.yaml                   # see template printed below

Usage:
    python scripts/train_helmet.py --data datasets/helmet/data.yaml --epochs 80
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DATA_YAML_TEMPLATE = """\
# Save to datasets/helmet/data.yaml
path: ./datasets/helmet
train: images/train
val: images/val
names:
  0: helmet
  1: no_helmet
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="helmet")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print("data.yaml not found. Template:")
        print(DATA_YAML_TEMPLATE)
        sys.exit(1)

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)
    print("Training complete. Best weights: runs/detect/%s/weights/best.pt" % args.name)
    print("Copy that file to ./models/helmet.pt to enable the helmet rule.")


if __name__ == "__main__":
    main()
