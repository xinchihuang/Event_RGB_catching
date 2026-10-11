"""Re-fit old Model tracks and compare against the December ground truth.

Pairing is an explicit assumption: old local ID i matches new case-local ID i.
Uses the historical fitter unchanged and the current plotter's GT offsets.
"""
import contextlib
import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import stage_path, resolve_path, setting_path
import csv
import io
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "old_model_new_gt_plots"
CAMERA = dict(fx=987.1678420424932, fy=986.0683237433913,
              cx=513.7733221750243, cy=739.7418995490075,
              dist_coeffs=[-0.11123899870851763, -0.020663002330730863,
                           -0.0002997445722290338, 0.0007964973013426446,
                           0.014818755754515196])


def legacy_options(default_output, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-source-root', help='Historical checkout containing fitting/ and utils.py')
    parser.add_argument('--tracks-dir', help='Historical track root containing ball_pixel_3D_records_model_*')
    parser.add_argument('--output-dir', default=str(default_output))
    args = parser.parse_args(argv)
    try:
        args.legacy_source_root = setting_path('legacy_source_root', args.legacy_source_root, 'LEGACY_SOURCE_ROOT')
        args.tracks_dir = setting_path('legacy_tracks', args.tracks_dir, 'LEGACY_TRACKS_DIR')
        args.output_dir = resolve_path(args.output_dir)
        for path in (args.legacy_source_root / 'fitting/parabola_fitter.py', args.legacy_source_root / 'utils.py'):
            if not path.is_file():
                raise FileNotFoundError(f'Missing historical source: {path}')
        if not args.tracks_dir.is_dir():
            raise FileNotFoundError(f'Missing historical tracks: {args.tracks_dir}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Historical source: {args.legacy_source_root}\nTracks: {args.tracks_dir}\nOutput: {args.output_dir}', flush=True)
    return args


def load_legacy_fitter(source_root):
    sys.path.insert(0, str(source_root))
    return (importlib.import_module('fitting.parabola_fitter').MonoRansac,
            importlib.import_module('utils').rotate_matrix_coordinate)


def main():
    args = legacy_options(OUTPUT)
    LEGACY, TRACKS, output = args.legacy_source_root, args.tracks_dir, args.output_dir
    MonoRansac, rotate_matrix_coordinate = load_legacy_fitter(LEGACY)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    # labels_12162025_README.txt: Excel ID = scene offset + local ID (0..99).
    # Keep this a one-way addition; do not wrap IDs with modulo or remap them.
    for task, case, start in [("normal", "l", 0), ("clutter", "c", 100),
                              ("dark", "d", 300), ("dark_clutter", "dc", 200)]:
        labels = np.loadtxt(ROOT / f"{task}.csv", delimiter=",")
        assert labels.shape == (100, 2)
        rows = []
        for i, label in enumerate(labels):
            path = TRACKS / f"ball_pixel_3D_records_model_{case}" / f"{i}.csv"
            row = dict(local_id=i, excel_id=start+i, source=str(path),
                       gt_x_m=label[0], gt_y_m=label[1], status="")
            rows.append(row)
            if not np.isfinite(label).all():
                row['status'] = 'missing_ground_truth'
                continue
            row.update(gt_Y=label[0]-0.05, gt_Z=label[1]+0.95)
            if not path.exists():
                row['status'] = 'missing_track'
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    points = np.loadtxt(path, delimiter=",", ndmin=2)
                if not points.size:
                    row['status'] = 'empty_track'
                    continue
                if points.shape[1] != 4:
                    raise ValueError('Expected four columns')
                row['observations_total'] = len(points)
                points[:, 3] -= points[0, 3]
                points = points[:10].copy()
                row['observations_used'] = len(points)
                if len(points) < 2 or len(np.unique(points[:, 3])) < 2:
                    row['status'] = 'insufficient_observations'
                    continue
                if not np.isfinite(points).all() or (points[:, 2] <= 0).any():
                    row['status'] = 'invalid_observations'
                    continue
                np.random.seed(20251216 + start + i)
                fitter = MonoRansac(CAMERA, rotate_matrix_coordinate(0, 45, -90),
                                    rotate_matrix_coordinate(45, 0, 90))
                fitter.set_data(points)
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    fitted = fitter.get_params()
                    if np.count_nonzero(fitted['inliers']) < 2:
                        raise ValueError('Fewer than two fitted inliers')
                    if abs(fitted['params']['vx']) < 1e-10:
                        raise ValueError('Zero fitted vx')
                    prediction = np.asarray(fitter.predict())
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite prediction')
                error = float(np.linalg.norm(prediction[1:] - [row['gt_Y'], row['gt_Z']]))
                row.update(pred_Y=prediction[1], pred_Z=prediction[2], error_m=error,
                           success=error <= 0.25, status='evaluated')
            except Exception as exc:
                row.update(status='fit_failed', reason=f'{type(exc).__name__}: {exc}')

        fields = ['local_id', 'excel_id', 'source', 'status', 'reason', 'observations_total',
                  'observations_used', 'gt_x_m', 'gt_y_m', 'gt_Y', 'gt_Z',
                  'pred_Y', 'pred_Z', 'error_m', 'success']
        with (output / f'Model_{task}_details.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        evaluated = [r for r in rows if r['status'] == 'evaluated']
        success = sum(r['success'] for r in evaluated)
        outside = sum(not (-1 <= r['pred_Y'] <= 1 and 0 <= r['pred_Z'] <= 2)
                      for r in evaluated)
        stats = dict(case=case, task=task, evaluated=len(evaluated), success=success,
                     failed=len(evaluated)-success, skipped=100-len(evaluated),
                     outside_plot=outside,
                     success_rate=100*success/len(evaluated) if evaluated else None)
        summary.append(stats)
        fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
        for ok, color, name in [(True, 'green', 'Error <= 0.25 m'),
                                 (False, 'red', 'Error > 0.25 m')]:
            selected = [r for r in evaluated if r['success'] == ok]
            ax.scatter([r['pred_Y'] for r in selected], [r['pred_Z'] for r in selected],
                       c=color, s=65, alpha=.85, label=f'{name} (n={len(selected)})')
        ax.set(xlim=(-1, 1), ylim=(0, 2), xlabel='Y Coordinate (m)',
               ylabel='Z Coordinate (m)', title=f'Model — {case} / {task}\n'
               f'Old tracks + new ground truth | Success: {stats["success_rate"]:.1f}%')
        ax.set_aspect('equal')
        ax.grid(alpha=.3)
        ax.legend(loc='upper right')
        ax.text(.02, .02, f'Evaluated: {len(evaluated)} | Skipped: {stats["skipped"]}\n'
                f'Outside plot: {outside} (included in statistics)\n'
                'Assumption: matching local IDs across datasets\n'
                'GT offset: Y = x_m - 0.05; Z = y_m + 0.95',
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(facecolor='white', alpha=.9, edgecolor='#dddddd'))
        fig.savefig(output / f'landing_point_simple_Model_{task}.png', dpi=200)
        plt.close(fig)
        print(json.dumps(stats), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'method.txt').write_text(
        'Model old/test trajectories, first up to 10 observations; time rebased to zero.\n'
        f'Legacy MonoRansac imported from {LEGACY}. Ball diameter 0.104 m.\n'
        'Seed per sample = 20251216 + Excel ID. Legacy fitter and predict logic retained.\n'
        'Pairing assumes old local ID i matches new case-local ID i; not independently verified.\n'
        'Excel ranges: l 0-99, c 100-199, d 300-399, dc 200-299.\n'
        'GT mapping follows existing simple/multi plot: Y=x_m-0.05, Z=y_m+0.95.\n'
        'Success: YZ distance <= 0.25 m; denominator includes evaluated pairs only.\n'
        'Missing/invalid GT, tracks, and failed fits excluded and logged. No error outlier filtering.\n'
        'Original plot limits Y [-1,1], Z [0,2]; off-plot predictions remain in statistics.\n')


if __name__ == '__main__':
    main()
