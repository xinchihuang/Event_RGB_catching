"""Compare saved landing-point errors by scene and method, without refitting."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import stage_path, resolve_path, setting_path
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from plot_landing_prediction_simple import POSITION_CONFIG, build_real_landing

SCENES = [('normal', 'Normal'), ('clutter', 'Clutter'),
          ('dark', 'Dark'), ('dark_clutter', 'Dark + clutter')]
METHODS = ['Model', 'Event', 'RGB']
COLORS = dict(Model='#d62728', Event='#1f77b4', RGB='#2ca02c')
MAX_ERROR_M = 0.80


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path,
                        default=None)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'landing_error_boxplot')
    args = parser.parse_args()
    try:
        args.root_dir = stage_path('evaluation', explicit=args.root_dir)
        args.output_dir = resolve_path(args.output_dir)
        if not args.root_dir.is_dir():
            raise FileNotFoundError(f'Missing evaluation input: {args.root_dir}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Evaluation input: {args.root_dir}\nPlot output: {args.output_dir}', flush=True)
    root = args.root_dir.resolve()
    groups, details, summaries, hashes = {}, [], [], {}
    for task, _ in SCENES:
        label_path = root / f'{task}.csv'
        labels = np.loadtxt(label_path, delimiter=',', ndmin=2)
        if labels.shape[1] < 2:
            raise ValueError(f'Expected coordinate labels: {label_path}')
        hashes[str(label_path)] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        for method in METHODS:
            errors, counts = [], Counter()
            for index, label in enumerate(labels):
                path = root / f'{method}_{task}/ball_pixel_3D_records/{index}.csv'
                row = dict(method=method, scene=task, local_id=index,
                           source=str(path), status='', error_m='', gt_y_m='', gt_z_m='')
                truth = build_real_landing(label)
                if truth is None:
                    row['status'] = 'invalid_label'
                else:
                    row.update(gt_y_m=truth[1], gt_z_m=truth[2])
                    if not path.is_file():
                        row['status'] = 'missing_prediction'
                    else:
                        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                        try:
                            prediction = np.loadtxt(path, delimiter=',')
                        except ValueError:
                            prediction = np.array([])
                        if prediction.shape != (3,) or not np.isfinite(prediction).all():
                            row['status'] = 'invalid_prediction'
                        elif np.allclose(prediction, 0):
                            row['status'] = 'zero_prediction'
                        else:
                            error = float(np.linalg.norm(prediction[1:] - np.asarray(truth)[1:]))
                            row['error_m'] = error
                            if error > MAX_ERROR_M:
                                row['status'] = 'above_80cm'
                            else:
                                errors.append(error)
                                row['status'] = 'valid'
                counts[row['status']] += 1
                details.append(row)
            groups[task, method] = errors
            summaries.append(dict(
                scene=task, method=method, label_rows=len(labels), valid=len(errors),
                zero_prediction=counts['zero_prediction'],
                above_80cm=counts['above_80cm'],
                missing_prediction=counts['missing_prediction'],
                invalid_prediction=counts['invalid_prediction'], invalid_label=counts['invalid_label'],
                median_m=float(np.median(errors)) if errors else '',
                mean_m=float(np.mean(errors)) if errors else '',
                q1_m=float(np.percentile(errors, 25)) if errors else '',
                q3_m=float(np.percentile(errors, 75)) if errors else '',
                max_m=max(errors) if errors else '',
                within_25cm=sum(e <= POSITION_CONFIG['success_threshold'] for e in errors)))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_groups(groups, output)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'sample_errors.csv', details)
    (output / 'provenance.json').write_text(json.dumps(dict(
        source_root=str(root), position_config=POSITION_CONFIG,
        error_definition='Euclidean distance in the YZ plane, metres in CSV; centimetres in figure',
        exclusions=['zero_prediction', 'missing_prediction', 'invalid_prediction', 'invalid_label', 'above_80cm'],
        max_error_m=MAX_ERROR_M,
        refit=False, source_sha256=hashes,
        normal_label_note='Uses the original results-root normal.csv, including its local ID 12 label.'),
        indent=2, ensure_ascii=False))
    print(json.dumps(summaries, indent=2))
    print(f'Saved to {output}')


def plot_groups(groups, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 5))
    maximum = max((e for errors in groups.values() for e in errors), default=.25) * 100
    for scene_index, (task, _) in enumerate(SCENES):
        center = scene_index * 4
        if scene_index % 2 == 0:
            ax.axvspan(center - 1.6, center + 1.6, color='#f3f6fa', zorder=0)
        for method_index, method in enumerate(METHODS):
            x = center + (method_index - 1) * .85
            values = np.asarray(groups[task, method]) * 100
            color = COLORS[method]
            ax.text(x, 1.018, f'n={len(values)}', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=10, color=color)
            if not len(values):
                ax.text(x, 3, 'No valid\nprediction', ha='center', va='bottom',
                        fontsize=7.5, color=color)
                continue
            ax.boxplot([values], positions=[x], widths=.59, patch_artist=True,
                       manage_ticks=False, whis=1.5, showfliers=True,
                       boxprops=dict(facecolor=color, edgecolor=color, alpha=.72),
                       medianprops=dict(color='#172333', linewidth=2),
                       whiskerprops=dict(color=color, linewidth=1.4),
                       capprops=dict(color=color, linewidth=1.4),
                       flierprops=dict(marker='o', markersize=4, markerfacecolor='none',
                                       markeredgecolor=color, alpha=.75))
    threshold = POSITION_CONFIG['success_threshold'] * 100
    ax.axhline(threshold, color='#b84242', linestyle='--', linewidth=1.4)
    ax.text(13.4, threshold + 1.5, '25 cm threshold', color='#b84242',
            ha='right', fontsize=10)
    ax.set(xlim=(-1.65, 13.65), ylim=(0, max(40, maximum * 1.08)),
           ylabel='Landing-point error (cm)', xlabel='Scene')
    ax.set_xticks([i * 4 for i in range(len(SCENES))])
    ax.set_xticklabels([title for _, title in SCENES], fontsize=13)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=COLORS[m], label=m, alpha=.72) for m in METHODS],
              loc='upper left', ncol=3, frameon=False)
    fig.subplots_adjust(left=.075, right=.99, bottom=.12, top=.94)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'landing_error_boxplot.{suffix}', dpi=220,
                    bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


if __name__ == '__main__':
    main()
