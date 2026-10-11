"""Run the legacy trajectory fitter and original plot_accuracy drawing function."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_old_model_new_gt import CAMERA, ROOT, legacy_options, load_legacy_fitter


def main():
    args = legacy_options(ROOT / 'ground_truth_plots')
    LEGACY, TRACKS, output = args.legacy_source_root, args.tracks_dir, args.output_dir
    MonoRansac, rotate_matrix_coordinate = load_legacy_fitter(LEGACY)
    spec = importlib.util.spec_from_file_location('legacy_plot_accuracy',
                                                 LEGACY / 'evaluation_tools/plot_accuracy.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case, offset in [('l', 0), ('c', 100), ('d', 200), ('dc', 300)]:
        labels = np.loadtxt(LEGACY / f'evaluation_tools/label_{case}.csv', delimiter=',')
        results, audit = [], []
        for i, label in enumerate(labels):
            row = dict(id=i, grid_label=label, status='')
            audit.append(row)
            if label == 0 or not np.isfinite(label):
                row['status'] = 'invalid_label'
                continue
            source = TRACKS / f'ball_pixel_3D_records_model_{case}/{i}.csv'
            if not source.exists():
                row['status'] = 'missing_track'
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    observations = np.loadtxt(source, delimiter=',', ndmin=2)
                if not observations.size or len(observations) < 2:
                    row['status'] = 'empty_or_insufficient_track'
                    continue
                observations[:, 3] -= observations[0, 3]
                observations = observations[:10].copy()
                if not np.isfinite(observations).all() or (observations[:, 2] <= 0).any():
                    row['status'] = 'invalid_observations'
                    continue
                np.random.seed(20251216 + offset + i)
                fitter = MonoRansac(CAMERA, rotate_matrix_coordinate(0, 45, -90),
                                    rotate_matrix_coordinate(45, 0, 90))
                fitter.set_data(observations)
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    fit = fitter.get_params()
                    if np.count_nonzero(fit['inliers']) < 2 or abs(fit['params']['vx']) < 1e-10:
                        raise ValueError('Invalid fitted parameters or inliers')
                    prediction = np.asarray(fitter.predict())
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite prediction')
                truth = legacy.index_to_position(label)
                error = float(np.linalg.norm(prediction[1:] - np.asarray(truth)[1:]))
                row.update(status='evaluated', pred_Y=prediction[1], pred_Z=prediction[2],
                           gt_Y=truth[1], gt_Z=truth[2], error_m=error, success=error<=.25)
                results.append(dict(impact=prediction, label=truth, trajectory_id=i))
            except Exception as exc:
                row.update(status='fit_failed', reason=str(exc))
        case_dir = output / f'legacy_Model_{case}'
        case_dir.mkdir(exist_ok=True)
        previous = Path.cwd()
        try:
            os.chdir(case_dir)
            legacy.plot_2d_impact_comparison(results, success_threshold=.25)
            plt.close('all')
        finally:
            os.chdir(previous)
        with (case_dir / 'details.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id','grid_label','status','reason',
                                                   'pred_Y','pred_Z','gt_Y','gt_Z','error_m','success'])
            writer.writeheader()
            writer.writerows(audit)
        good = [r for r in audit if r['status']=='evaluated']
        summary = dict(case=case, total_labels=len(labels), evaluated=len(good),
                       success=sum(r['success'] for r in good), skipped=len(labels)-len(good))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (output / 'legacy_summary.json').write_text(json.dumps(summaries, indent=2))
    (output / 'legacy_method.txt').write_text(
        f'Original plot_accuracy.py drawing function and index_to_position from {LEGACY}.\n'
        'Uses legacy label_l/c/d/dc.csv, not the new Excel coordinates. Model old/test tracks.\n'
        'Same first up to 10 observations, t rebased to zero, MonoRansac; seed 20251216+case_offset+ID.\n'
        'Legacy label conversion offsets: y_bias=-0.05, z_bias=-0.2, plus 0.05 grid-center offsets.\n'
        'Threshold 0.25 m. Missing/invalid inputs and failed fits skipped; see details.csv.\n'
        'Original main hardcodes an unavailable data path; this runner supplies the existing old/test paths.\n')


if __name__ == '__main__':
    main()
