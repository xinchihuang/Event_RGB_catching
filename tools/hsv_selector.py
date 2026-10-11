import cv2
import numpy as np
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------
# Utils
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def hsv_range_around(hsv, dh=10, ds=40, dv=40):
    h, s, v = map(int, hsv)
    lower = (clamp(h - dh, 0, 179), clamp(s - ds, 0, 255), clamp(v - dv, 0, 255))
    upper = (clamp(h + dh, 0, 179), clamp(s + ds, 0, 255), clamp(v + dv, 0, 255))
    return lower, upper

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="tracking/l.jpg",
                        help="Image path or camera index (project-relative; default: tracking/l.jpg)")
    parser.add_argument("--x", type=int, default=None, help="Optional pixel x")
    parser.add_argument("--y", type=int, default=None, help="Optional pixel y")
    parser.add_argument("--dh", type=int, default=10)
    parser.add_argument("--ds", type=int, default=40)
    parser.add_argument("--dv", type=int, default=40)
    args = parser.parse_args()

    # Resolve image paths relative to the project; numeric sources select a camera.
    source = args.source

    cap = None

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index={source}")
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read camera frame")
        mode = "video"
    else:
        source = str((PROJECT_ROOT / Path(source).expanduser()).resolve())
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {source}")
        mode = "image"

    win = "HSV Picker (click to sample, ESC to quit)"
    last_click = {"pt": None}

    def report_point(img, x, y):
        h, w = img.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            print(f"[!] Point ({x},{y}) out of bounds (w={w}, h={h})")
            return

        bgr = img[y, x].tolist()
        hsv = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0, 0].tolist()

        print(f"\nPoint (x={x}, y={y})")
        print(f"  BGR = {bgr}")
        print(f"  HSV = {hsv}")

        lo, hi = hsv_range_around(hsv, args.dh, args.ds, args.dv)
        print(f"  Suggested HSV range: lower={lo}, upper={hi}")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click["pt"] = (x, y)
            report_point(param, x, y)

    # If explicit coordinates are provided, print once and exit
    if args.x is not None and args.y is not None and mode == "image":
        report_point(frame, args.x, args.y)
        return

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if mode == "video":
            ret, frame = cap.read()
            if not ret:
                break

        vis = frame.copy()
        if last_click["pt"] is not None:
            cv2.drawMarker(vis, last_click["pt"], (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        cv2.setMouseCallback(win, on_mouse, frame)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1 if mode == "video" else 30) & 0xFF
        if key == 27:
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
