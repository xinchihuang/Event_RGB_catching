import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def read_prediction_csv(csv_path):
    """
    Read landing prediction from CSV file.
    Expected format: single line with x, y, z coordinates.
    """
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
        # Drop rows that are entirely NaN (e.g., blank lines).
        valid_rows = ~np.isnan(labels).all(axis=1)
        labels = labels[valid_rows]
        if labels.size == 0:
            print(f"Warning: No valid label rows in {label_file}")
            return None
        if labels.shape[1] >= 2:
            return labels
        print(f"Error: Invalid label format in {label_file}. Expected at least 2 columns.")
        return None

    print(f"Error: Invalid label format in {label_file}.")
    return None


def plot_2d_prediction_comparison(exp_results, output_path, title_prefix,
                                  success_threshold=0.25, show=False,
                                  unsuccess_output_path=None):
    """
    Plot 2D comparison of real vs predicted landing points.
    """
    real_y = []
    real_z = []
    pred_y = []
    pred_z = []
    success_flags = []
    trajectory_ids = []

    for result in exp_results:
        real_landing = result["label"]
        pred_landing = result["prediction"]
        traj_id = result["trajectory_id"]

        real_y.append(real_landing[1])
        real_z.append(real_landing[2])
        pred_y.append(pred_landing[1])
        pred_z.append(pred_landing[2])
        trajectory_ids.append(traj_id)

        distance = np.sqrt((real_landing[1] - pred_landing[1]) ** 2 +
                           (real_landing[2] - pred_landing[2]) ** 2)
        success_flags.append(distance <= success_threshold)

    real_y = np.array(real_y)
    real_z = np.array(real_z)
    pred_y = np.array(pred_y)
    pred_z = np.array(pred_z)
    success_flags = np.array(success_flags)

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 14})

    successful_mask = success_flags
    plt.scatter(real_y[successful_mask], real_z[successful_mask],
                c="green", marker="o", s=100, alpha=0.7, label="Real (Success)")
    plt.scatter(pred_y[successful_mask], pred_z[successful_mask],
                c="lightgreen", marker="x", s=100, alpha=0.7, label="Predicted (Success)")

    failed_mask = ~success_flags
    plt.scatter(real_y[failed_mask], real_z[failed_mask],
                c="red", marker="o", s=100, alpha=0.7, label="Real (Failed)")
    plt.scatter(pred_y[failed_mask], pred_z[failed_mask],
                c="pink", marker="x", s=100, alpha=0.7, label="Predicted (Failed)")

    for i in range(len(real_y)):
        color = "green" if success_flags[i] else "red"
        plt.plot([real_y[i], pred_y[i]], [real_z[i], pred_z[i]],
                 color=color, alpha=0.3, linewidth=1)

    for i in range(len(real_y)):
        if not success_flags[i]:
            plt.annotate(f"{trajectory_ids[i]}",
                         (real_y[i], real_z[i]),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=8, alpha=0.7)

    success_rate = (np.sum(success_flags) / len(success_flags) * 100) if len(success_flags) > 0 else 0
    errors = np.sqrt((real_y - pred_y) ** 2 + (real_z - pred_z) ** 2)
    mean_error = float(np.mean(errors)) if len(errors) > 0 else 0

    if unsuccess_output_path:
        failed_rows = []
        for i in range(len(exp_results)):
            if success_flags[i]:
                continue
            failed_rows.append(trajectory_ids[i])
        if failed_rows:
            os.makedirs(os.path.dirname(unsuccess_output_path) or ".", exist_ok=True)
            with open(unsuccess_output_path, "w", encoding="utf-8") as handle:
                handle.write("trajectory_id\n")
                for row in failed_rows:
                    handle.write(f"{row}\n")

    plt.xlabel("Y Coordinate (m)")
    plt.ylabel("Z Coordinate (m)")
    plt.title(f"Landing Point Prediction Results - {title_prefix}\n"
              f"Success Rate: {success_rate:.1f}% | Mean Error: {mean_error:.3f}m | "
              f"Threshold: {success_threshold}m")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xlim(-1, 1)
    plt.ylim(0, 2)

    stats_text = f"Total Trajectories: {len(exp_results)}\n" \
                 f"Successful: {np.sum(success_flags)}\n" \
                 f"Failed: {np.sum(~success_flags)}\n" \
                 f"Success Rate: {success_rate:.1f}%\n" \
                 f"Mean Error: {mean_error:.3f}m"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    if show:
        plt.show()
    plt.close()

    print(f"\n=== Landing Point Prediction Results ({title_prefix}) ===")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Error: {mean_error:.3f}m")
    print(f"Success Threshold: {success_threshold}m")
    print(f"Total Trajectories: {len(exp_results)}")
    print(f"Successful Predictions: {np.sum(success_flags)}")
    print(f"Failed Predictions: {np.sum(~success_flags)}")
    print(f"Saved plot: {output_path}")
    if unsuccess_output_path and os.path.exists(unsuccess_output_path):
        print(f"Saved unsuccess cases: {unsuccess_output_path}")


def discover_combinations(root_dir):
    combos = []
    try:
        entries = os.listdir(root_dir)
    except Exception as exc:
        print(f"Error reading root directory {root_dir}: {exc}")
        return combos

    for name in entries:
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        if "_" not in name:
            continue
        test_mode, task = name.split("_", 1)
        combos.append((test_mode, task, path))
    return combos


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
        return label_map[mode_task_key]
    if task in label_map:
        return label_map[task]
    if default_path:
        return default_path
    return os.path.join(root_dir, f"{task}.csv")


def main():
    parser = argparse.ArgumentParser(description="Plot landing prediction results for multiple test modes and tasks.")
    parser.add_argument("--root-dir", type=str,
                        default="/media/xinchi/data/test_data/evaluation_results",
                        help="Root directory containing {test_mode}_{task} folders.")
    parser.add_argument("--test-modes", type=str, default="Event,RGB,Model",
                        help="Comma-separated test modes to analyze (e.g., Event,RGB,Model).")
    parser.add_argument("--tasks", type=str, default="normal,dark,clutter,dark_clutter",
                        help="Comma-separated tasks to analyze (e.g., normal,dark,clutter,dark_clutter).")
    parser.add_argument("--prediction-subdir", type=str, default="ball_pixel_3D_records",
                        help="Subdirectory under each mode/task folder that stores prediction CSVs.")
    parser.add_argument("--label-file", action="append", default=['normal.csv','dark.csv','clutter.csv','dark_clutter.csv'],
                        help="Label file mapping: KEY=PATH. KEY can be task or test_mode_task. Repeatable.")
    parser.add_argument("--label-default", type=str, default="",
                        help="Default label file path used when no specific mapping is provided. "
                             "If empty, uses {root-dir}/{task}.csv.")
    parser.add_argument("--success-threshold", type=float, default=0.25,
                        help="Distance threshold for successful prediction (meters).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save plots.")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show plots interactively.")
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory {args.root_dir} does not exist.")
        return

    test_modes = parse_list(args.test_modes)
    tasks = parse_list(args.tasks)
    label_map = parse_label_map(args.label_file)

    if test_modes and tasks:
        combos = []
        for test_mode in test_modes:
            for task in tasks:
                base_dir = os.path.join(args.root_dir, f"{test_mode}_{task}")
                combos.append((test_mode, task, base_dir))
    else:
        combos = discover_combinations(args.root_dir)
        if test_modes:
            combos = [c for c in combos if c[0] in test_modes]
        if tasks:
            combos = [c for c in combos if c[1] in tasks]

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
            print(f"Skipping {test_mode}_{task}: label file not set. Use --label-file or --label-default.")
            continue
        if not os.path.exists(label_path):
            print(f"Skipping {test_mode}_{task}: label file {label_path} does not exist.")
            continue

        labels = load_labels(label_path)
        if labels is None:
            continue

        exp_results = []
        for i in range(len(labels)):
            label_row = labels[i]
            if label_row.size < 2:
                continue
            # Label rows use the last two columns as real y/z; convert to camera z.
            y_real = float(label_row[-2])
            z_real = float(label_row[-1])
            if not np.isfinite(y_real) or not np.isfinite(z_real):
                continue
            if np.allclose([y_real, z_real], 0):
                continue
            y_c = y_real -0.05
            z_c = z_real + 0.95
            x_c = float(label_row[0]) if label_row.size >= 3 else 0.0
            real_landing = [x_c, y_c, z_c]

            csv_path = os.path.join(prediction_dir, f"{i}.csv")
            if not os.path.exists(csv_path):
                continue

            prediction = read_prediction_csv(csv_path)
            if prediction is None:
                continue

            exp_results.append({
                "prediction": prediction,
                "label": real_landing,
                "trajectory_id": i,
            })

        if not exp_results:
            print(f"No valid results found for {test_mode}_{task}.")
            continue

        output_name = f"landing_point_comparison_{test_mode}_{task}.png"
        output_path = os.path.join(args.output_dir, output_name)
        unsuccess_name = f"unsuccess_cases_{test_mode}_{task}.csv"
        unsuccess_path = os.path.join(args.output_dir, unsuccess_name)
        title_prefix = f"{test_mode}_{task}"
        plot_2d_prediction_comparison(
            exp_results,
            output_path=output_path,
            title_prefix=title_prefix,
            success_threshold=args.success_threshold,
            unsuccess_output_path=unsuccess_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
