#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import stage_path, resolve_path, setting_path
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import get_config
from utils import pixel_to_world, rotate_matrix_coordinate


DEFAULT_INTRINSICS = {
    "fx": 987.1678420424932,
    "fy": 986.0683237433913,
    "cx": 513.7733221750243,
    "cy": 739.7418995490075,
}
DEFAULT_DIST = [
    -0.11123899870851763,
    -0.020663002330730863,
    -0.0002997445722290338,
    0.0007964973013426446,
    0.014818755754515196,
]
DISPLAY_X_RANGE = (0.0, 3.0)
DISPLAY_Y_RANGE = (-1.5, 1.5)
DISPLAY_Z_RANGE = (0.0, 2.0)


def load_csv(path: str) -> np.ndarray:
    rows = []
    bad_rows = 0
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) != 4:
                bad_rows += 1
                continue
            try:
                values = [float(x) for x in row]
            except ValueError:
                bad_rows += 1
                continue
            rows.append(values)
    if bad_rows:
        print(f"Skipped {bad_rows} invalid rows.")
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def set_display_range(ax) -> None:
    ax.set_xlim(*DISPLAY_X_RANGE)
    ax.set_ylim(*DISPLAY_Y_RANGE)
    ax.set_zlim(*DISPLAY_Z_RANGE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D points from a single CSV of (x, y, width, t)."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to a single CSV file.",
    )
    parser.add_argument(
        "--ball-diameter",
        type=float,
        default=0.104,
        help="Ball diameter in meters (default: 0.104).",
    )
    parser.add_argument(
        "--save",
        default="results.png",
        help="Output path to save plot (PNG). If empty, show interactively.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable the interactive plot and render a static figure.",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        default=0.0,
        help="Show points within the last N seconds from the slider value (default: 0 = cumulative).",
    )
    args = parser.parse_args()
    args.csv_path = str(resolve_path(args.csv_path) if args.csv_path is not None else
                        stage_path('evaluation') / 'Event_normal/ball_pixel_3D_records_ball/5.csv')
    if args.save:
        args.save = str(resolve_path(args.save))

    csv_path = args.csv_path
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    data = load_csv(csv_path)
    if data.size == 0:
        print("No valid rows found.")
        return 1

    config_loader = get_config()
    intrinsics = config_loader.get("camera_intrinsics", {}) or {}
    dist = intrinsics.get("distortion_coeffs", DEFAULT_DIST) or DEFAULT_DIST
    dist = list(dist)
    if len(dist) < 5:
        dist = dist + [0.0] * (5 - len(dist))
    if len(dist) > 5:
        dist = dist[:5]

    camera_params = {
        "fx": intrinsics.get("fx", DEFAULT_INTRINSICS["fx"]),
        "fy": intrinsics.get("fy", DEFAULT_INTRINSICS["fy"]),
        "cx": intrinsics.get("cx", DEFAULT_INTRINSICS["cx"]),
        "cy": intrinsics.get("cy", DEFAULT_INTRINSICS["cy"]),
        "dist_coeffs": np.array(dist, dtype=np.float64),
    }
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_back = rotate_matrix_coordinate(
        camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2]
    )

    world = pixel_to_world(
        data, camera_params, rotation_matrix_back, ball_diameter=args.ball_diameter
    )
    t = data[:, 3]
    valid = np.isfinite(world).all(axis=1)
    if not np.any(valid):
        print("No valid points after filtering.")
        return 1
    x = world[:, 0]
    y = world[:, 1]
    z = world[:, 2]
    x = x[valid]
    y = y[valid]
    z = z[valid]
    t = t[valid]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        x,
        y,
        z,
        c=t,
        s=8,
        cmap="viridis",
        alpha=0.9,
    )
    fig.colorbar(sc, ax=ax, label="time (s)")
    ax.set_title("Reconstructed 3D points (world coords)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_display_range(ax)

    if not args.no_interactive:
        t_min = float(np.nanmin(t))
        t_max = float(np.nanmax(t))
        sc.set_clim(t_min, t_max)
        fig.subplots_adjust(bottom=0.2)
        slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
        time_slider = Slider(slider_ax, "time", t_min, t_max, valinit=t_min)

        window = max(args.time_window, 0.0)

        def update(val: float) -> None:
            if window > 0.0:
                mask = (t >= val - window) & (t <= val)
            else:
                mask = t <= val
            sc._offsets3d = (x[mask], y[mask], z[mask])
            sc.set_array(t[mask])
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        update(t_min)

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {args.save}")
        plt.show()
        return 0

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
