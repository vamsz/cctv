"""Camera calibration helper.

Grabs a single frame from a camera source, lets you click two points to
define the stop line, then drag a rectangle for the signal ROI, and
prints the YAML snippet you should paste into config/cameras.yaml.

Usage:
    python scripts/calibrate.py --source rtsp://...     [--out config/CAM01.yaml]
    python scripts/calibrate.py --source ./sample.mp4

Controls:
    Left-click two points  -> stop line
    Drag right-click       -> signal ROI rectangle
    Press 'd' + arrow keys -> nudge the lawful direction unit vector
    Press 's'              -> print YAML & save snapshot to ./data/calibration/
    Press 'q'              -> quit
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def grab_frame(source: str):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Could not open {source}")
        sys.exit(1)
    # Drain a few frames so RTSP keyframes settle.
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    cap.release()
    if frame is None:
        print("No frame received.")
        sys.exit(1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--id", default="CAM01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frame = grab_frame(args.source)
    H, W = frame.shape[:2]
    print(f"Frame: {W}x{H}")

    state = {
        "stop_line": [],            # up to 2 points
        "signal_drag_start": None,
        "signal_roi": None,         # (x1,y1,x2,y2)
        "direction": [0.0, -1.0],
    }

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state["stop_line"]) >= 2:
                state["stop_line"] = []
            state["stop_line"].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["signal_drag_start"] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and state["signal_drag_start"]:
            x1, y1 = state["signal_drag_start"]
            state["signal_roi"] = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            state["signal_drag_start"] = None

    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        view = frame.copy()
        if len(state["stop_line"]) == 2:
            cv2.line(view, state["stop_line"][0], state["stop_line"][1], (0, 0, 255), 2)
        for p in state["stop_line"]:
            cv2.circle(view, p, 5, (0, 0, 255), -1)
        if state["signal_roi"]:
            x1, y1, x2, y2 = state["signal_roi"]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dx, dy = state["direction"]
        cx, cy = W // 2, H // 2
        cv2.arrowedLine(view, (cx, cy), (cx + int(dx * 80), cy + int(dy * 80)), (0, 200, 255), 3, tipLength=0.25)
        cv2.putText(view, "L-click x2 = stop line | R-drag = signal | arrows = direction | s = save | q = quit",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)

        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            print_yaml(args.id, args.source, state)
            outdir = ROOT / "data" / "calibration"
            outdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(outdir / f"{args.id}_snapshot.jpg"), frame)
        elif k == 82:  # up
            state["direction"] = [state["direction"][0], state["direction"][1] - 0.1]
        elif k == 84:  # down
            state["direction"] = [state["direction"][0], state["direction"][1] + 0.1]
        elif k == 81:  # left
            state["direction"] = [state["direction"][0] - 0.1, state["direction"][1]]
        elif k == 83:  # right
            state["direction"] = [state["direction"][0] + 0.1, state["direction"][1]]

    cv2.destroyAllWindows()


def print_yaml(cam_id, source, state):
    sl = state["stop_line"]
    roi = state["signal_roi"]
    if len(sl) != 2:
        print("Need exactly 2 stop-line points.")
        return
    if roi is None:
        print("Signal ROI not set; using None.")
    print("\n--- paste into config/cameras.yaml under `cameras:` ---")
    print(f"  - id: {cam_id}")
    print(f"    name: \"{cam_id}\"")
    print(f"    source: \"{source}\"")
    print(f"    fps_cap: 15")
    print(f"    stop_line: [[{sl[0][0]}, {sl[0][1]}], [{sl[1][0]}, {sl[1][1]}]]")
    if roi:
        print(f"    signal_roi: [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}]")
    print(f"    direction: [{state['direction'][0]:.2f}, {state['direction'][1]:.2f}]")
    print(f"    enabled: true\n")


if __name__ == "__main__":
    main()
