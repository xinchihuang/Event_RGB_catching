import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import stage_path, resolve_path, setting_path
import os

import matplotlib.pyplot as plt
import numpy as np


# Unified configuration for label-to-camera conversion and success threshold.
POSITION_CONFIG = {
    "y_offset": -0.05,
    "z_offset": 0.95,
    "success_threshold": 0.25,
}

PLOT_CONFIG = {
    "xlim": (-1, 1),
    "ylim": (0, 2),
    "figsize": (10, 8),
    "font_size": 14,
    "axis_label_size": 30,
    "tick_label_size": 24,
    "marker_size": 100,
    "x_margin_ratio": 0.005,
    "y_margin_ratio": 0.005,
    "layout_pad": 0.1,
}


def read_prediction_csv(csv_path):
    """Read a single landing prediction CSV with expected shape (3,)."""
    try:
        data = np.loadtxt(csv_path, delimiter=",")
        if data.size == 0:
            print(f"Warning: Empty CSV file {csv_path}")
            return None
        if data.ndim == 1 and len(data) == 3:
            return data
        if data.ndim == 2 and data.shape[1] == 3:
            print(f"Warning: Multiple rows in {csv_path}, using first row")
            return data[0]
        print(f"Error: Invalid data format in {csv_path}. Expected 3 columns (x,y,z)")
        return None
    except Exception as exc:
        print(f"Error reading {csv_path}: {exc}")
        return None


def load_labels(label_file):
    """Load labels and return labels_array (expects >=2 columns)."""
    try:
        labels = np.genfromtxt(label_file, delimiter=",", dtype=float)
    except Exception as exc:
        print(f"Error loading labels from {label_file}: {exc}")
        return None

    if labels.size == 0:
        print(f"Warning: Empty label file {label_file}")
        return None

    if labels.ndim == 1:
        if labels.size < 2:
            print(f"Error: Invalid label format in {label_file}. Expected at least 2 columns.")
            return None
        labels = labels.reshape(1, -1)

    if labels.ndim == 2:
        # Keep missing-label placeholders: row indices identify prediction files.
        if np.isnan(labels).all():
            print(f"Warning: No valid label rows in {label_file}")
            return None
        if labels.shape[1] >= 2:
            return labels
        print(f"Error: Invalid label format in {label_file}. Expected at least 2 columns.")
        return None

    print(f"Error: Invalid label format in {label_file}.")
    return None


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_label_map(items):
    label_map = {}
    for item in items:
        if "=" not in item:
            print(f"Warning: Invalid label mapping '{item}', expected KEY=PATH")
            continue
        key, path = item.split("=", 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            print(f"Warning: Invalid label mapping '{item}', expected KEY=PATH")
            continue
        label_map[key] = path
    return label_map


def resolve_label_path(test_mode, task, label_map, default_path, root_dir):
    mode_task_key = f"{test_mode}_{task}"
    if mode_task_key in label_map:
        return str(resolve_path(label_map[mode_task_key]))
    if task in label_map:
        return str(resolve_path(label_map[task]))
    if default_path:
        return str(resolve_path(default_path))
    return os.path.join(root_dir, f"{task}.csv")


def build_real_landing(label_row):
    """Convert a label row into camera-space landing coordinates."""
    y_real = float(label_row[-2])
    z_real = float(label_row[-1])
    if not np.isfinite(y_real) or not np.isfinite(z_real):
        return None
    if np.allclose([y_real, z_real], 0):
        return None

    y_c = y_real + POSITION_CONFIG["y_offset"]
    z_c = z_real + POSITION_CONFIG["z_offset"]
    x_c = float(label_row[0]) if label_row.size >= 3 else 0.0
    return [x_c, y_c, z_c]


def collect_results(prediction_dir, labels):
    results = []
    for i in range(len(labels)):
        label_row = labels[i]
        if label_row.size < 2:
            continue

        real_landing = build_real_landing(label_row)
        if real_landing is None:
            continue

        csv_path = os.path.join(prediction_dir, f"{i}.csv")
        if not os.path.exists(csv_path):
            continue

        prediction = read_prediction_csv(csv_path)
        if prediction is None:
            continue
        if np.allclose(prediction, [0.0, 0.0, 0.0]):
            continue

        results.append({
            "prediction": prediction,
            "label": real_landing,
        })
    return results


def plot_success_only(exp_results, output_path, title_prefix, show=False):
    """Plot predicted points only, colored by success/failure."""
    pred_success_y = []
    pred_success_z = []
    pred_fail_y = []
    pred_fail_z = []

    threshold = POSITION_CONFIG["success_threshold"]

    for result in exp_results:
        real_landing = result["label"]
        pred_landing = result["prediction"]
        distance = np.sqrt(
            (real_landing[1] - pred_landing[1]) ** 2 +
            (real_landing[2] - pred_landing[2]) ** 2
        )
        success = distance <= threshold

        if success:
            pred_success_y.append(pred_landing[1])
            pred_success_z.append(pred_landing[2])
        else:
            pred_fail_y.append(pred_landing[1])
            pred_fail_z.append(pred_landing[2])

    plt.figure(figsize=PLOT_CONFIG["figsize"])
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})

    plt.scatter(
        pred_success_y,
        pred_success_z,
        c="green",
        marker="o",
        s=PLOT_CONFIG["marker_size"],
        alpha=0.85,
    )
    plt.scatter(
        pred_fail_y,
        pred_fail_z,
        c="red",
        marker="o",
        s=PLOT_CONFIG["marker_size"],
        alpha=0.85,
    )

    plt.xlabel("Y Coordinate (m)", fontsize=PLOT_CONFIG["axis_label_size"])
    plt.ylabel("Z Coordinate (m)", fontsize=PLOT_CONFIG["axis_label_size"])
    plt.xticks(fontsize=PLOT_CONFIG["tick_label_size"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_label_size"])
    plt.grid(True, alpha=0.3)
    plt.xlim(*PLOT_CONFIG["xlim"])
    plt.ylim(*PLOT_CONFIG["ylim"])
    plt.margins(x=PLOT_CONFIG["x_margin_ratio"], y=PLOT_CONFIG["y_margin_ratio"])
    plt.tight_layout(pad=PLOT_CONFIG["layout_pad"])
    plt.savefig(output_path, dpi=300)

    if show:
        plt.show()
    plt.close()

    success_count = len(pred_success_y)
    fail_count = len(pred_fail_y)
    print(f"{title_prefix}: success={success_count}, failed={fail_count}, saved={output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple landing-point plot without connection lines or failure IDs.")
    parser.add_argument("--root-dir", type=str,
                        default=None,
                        help="Root directory containing {test_mode}_{task} folders.")
    parser.add_argument("--test-modes", type=str, default="Event,RGB,Model",
                        help="Comma-separated test modes to analyze.")
    parser.add_argument("--tasks", type=str, default="normal,dark,clutter,dark_clutter",
                        help="Comma-separated tasks to analyze.")
    parser.add_argument("--prediction-subdir", type=str, default="ball_pixel_3D_records",
                        help="Subdirectory under each mode/task folder that stores prediction CSVs.")
    parser.add_argument("--label-file", action="append", default=["normal.csv", "dark.csv", "clutter.csv", "dark_clutter.csv"],
                        help="Label file mapping: KEY=PATH. KEY can be task or test_mode_task. Repeatable.")
    parser.add_argument("--label-default", type=str, default="",
                        help="Default label file path used when no specific mapping is provided.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save plots.")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show plots interactively.")
    args = parser.parse_args()
    try:
        args.root_dir = stage_path('evaluation', explicit=args.root_dir)
        args.output_dir = resolve_path(args.output_dir)
        if not args.root_dir.is_dir():
            raise FileNotFoundError(f'Missing evaluation input: {args.root_dir}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Evaluation input: {args.root_dir}\nPlot output: {args.output_dir}', flush=True)

    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory {args.root_dir} does not exist.")
        return

    test_modes = parse_list(args.test_modes)
    tasks = parse_list(args.tasks)
    label_map = parse_label_map(args.label_file)

    combos = []
    for test_mode in test_modes:
        for task in tasks:
            base_dir = os.path.join(args.root_dir, f"{test_mode}_{task}")
            combos.append((test_mode, task, base_dir))

    if not combos:
        print("No matching test_mode/task combinations found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for test_mode, task, base_dir in combos:
        if not os.path.isdir(base_dir):
            print(f"Skipping {test_mode}_{task}: directory {base_dir} not found.")
            continue

        prediction_dir = os.path.join(base_dir, args.prediction_subdir)
        if not os.path.isdir(prediction_dir):
            print(f"Skipping {test_mode}_{task}: prediction directory {prediction_dir} not found.")
            continue

        label_path = resolve_label_path(test_mode, task, label_map, args.label_default, args.root_dir)
        if not label_path:
            print(f"Skipping {test_mode}_{task}: label file not set.")
            continue
        if not os.path.exists(label_path):
            print(f"Skipping {test_mode}_{task}: label file {label_path} does not exist.")
            continue

        labels = load_labels(label_path)
        if labels is None:
            continue

        exp_results = collect_results(prediction_dir, labels)
        if not exp_results:
            print(f"No valid results found for {test_mode}_{task}.")
            continue

        output_name = f"landing_point_simple_{test_mode}_{task}.png"
        output_path = os.path.join(args.output_dir, output_name)
        title_prefix = f"{test_mode}_{task}"
        plot_success_only(exp_results, output_path, title_prefix, show=args.show)


if __name__ == "__main__":
    main()
