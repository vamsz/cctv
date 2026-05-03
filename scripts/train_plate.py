"""Fine-tune a YOLO model for license plate detection (single class).

Dataset layout:
    datasets/plate/
        images/{train,val}/...jpg
        labels/{train,val}/...txt
        data.yaml

Usage:
    python scripts/train_plate.py --data datasets/plate/data.yaml --epochs 80
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DATA_YAML_TEMPLATE = """\
# Save to datasets/plate/data.yaml
path: ./datasets/plate
train: images/train
val: images/val
names:
  0: license_plate
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="plate")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print("data.yaml not found. Template:")
        print(DATA_YAML_TEMPLATE)
        sys.exit(1)

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)
    print("Best: runs/detect/%s/weights/best.pt -> copy to ./models/plate.pt" % args.name)


if __name__ == "__main__":
    main()
