"""NVDEC / OpenCV decode throughput benchmark.

Determines how many concurrent RTSP/file streams your GPU can decode
before falling behind real-time. Run this before deciding how many
camera pipelines to assign per GPU.

Usage:
    python scripts/benchmark_streams.py --source data/samples/traffic.mp4 --streams 1 4 8

The script reads each stream in a dedicated thread (mimicking the real
pipeline's StreamReader), decodes for 10 seconds, then reports:
  - Average decode FPS per stream
  - Total GPU decode throughput
  - Whether any stream fell below real-time (15 fps)
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path


def _decode_loop(source: str, duration: float, results: dict, idx: int) -> None:
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        results[idx] = {"fps": 0.0, "frames": 0, "error": "could not open"}
        return

    # Hint OpenCV to use NVDEC (GPU hardware decode) when available.
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    t0 = time.perf_counter()
    frames = 0
    errors = 0
    while time.perf_counter() - t0 < duration:
        ok, frame = cap.read()
        if not ok:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            errors += 1
            if errors > 10:
                break
            continue
        frames += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    results[idx] = {"fps": frames / elapsed if elapsed > 0 else 0.0, "frames": frames}


def benchmark(source: str, num_streams: int, duration: float = 10.0) -> None:
    print(f"\n{'─'*52}")
    print(f"  {num_streams} concurrent stream(s) × {duration:.0f} s  |  {source}")
    print(f"{'─'*52}")

    results: dict = {}
    threads = [
        threading.Thread(target=_decode_loop, args=(source, duration, results, i), daemon=True)
        for i in range(num_streams)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 5)

    total_fps = 0.0
    ok_count = 0
    for i, r in sorted(results.items()):
        fps = r.get("fps", 0.0)
        total_fps += fps
        status = "OK " if fps >= 14.0 else "SLOW"
        print(f"  Stream {i:02d}: {fps:6.1f} fps  [{status}]")
        if fps >= 14.0:
            ok_count += 1

    print(f"  ─────────────────────────────")
    print(f"  Total throughput : {total_fps:.1f} fps")
    print(f"  Real-time streams: {ok_count}/{num_streams}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode throughput benchmark")
    parser.add_argument("--source", required=True, help="Path to video file or rtsp:// URL")
    parser.add_argument(
        "--streams", nargs="+", type=int, default=[1, 2, 4, 6, 8],
        help="Number of concurrent streams to test (space-separated)"
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per test")
    args = parser.parse_args()

    if not (args.source.startswith("rtsp://") or Path(args.source).exists()):
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    print("\n=== CCTV Stream Decode Benchmark ===")
    try:
        import cv2
        print(f"OpenCV version  : {cv2.__version__}")
        build = cv2.getBuildInformation()
        nvdec_line = next((l for l in build.splitlines() if "NVDEC" in l or "CUDA" in l.upper()), "")
        if nvdec_line:
            print(f"CUDA/NVDEC info : {nvdec_line.strip()}")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU             : {props.name}  ({props.total_memory//1024**2} MB VRAM)")
    except Exception:
        pass

    for n in args.streams:
        benchmark(args.source, n, args.duration)

    print("\n=== Done ===")
    print("Rule of thumb: if a stream count falls below 14 fps, don't assign more cameras than that per GPU.")


if __name__ == "__main__":
    main()
