"""Export trained .pt weights to ONNX, then build a TensorRT engine.

Run on the deployment machine (the GPU model dictates the engine).

    python scripts/export_tensorrt.py --weights ./models/yolo11n.pt --imgsz 640 --half
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--int8", action="store_true", help="INT8 (needs calibration data)")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"missing: {weights}")
        sys.exit(1)

    model = YOLO(str(weights))
    # ONNX first (universal), then engine.
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, half=args.half, dynamic=False, simplify=True)
    print(f"ONNX: {onnx_path}")
    engine_path = model.export(format="engine", imgsz=args.imgsz, half=args.half, int8=args.int8)
    print(f"TensorRT engine: {engine_path}")
    print("Update DETECTOR_WEIGHTS / HELMET_WEIGHTS / PLATE_WEIGHTS in .env to point at the .engine file for max throughput.")


if __name__ == "__main__":
    main()
