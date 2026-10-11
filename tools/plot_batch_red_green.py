"""Use the original red/green plotter with ID-matched workbook ground truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from project_paths import resolve_path
sys.path.insert(0, str(PROJECT / 'evaluation_tools'))
import plot_landing_prediction_simple as original
from plot_replay_vs_excel import read_ground_truth, CASES, MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--xlsx', type=Path, required=True)
    args = parser.parse_args()
    run = resolve_path(args.run_dir)
    args.xlsx = resolve_path(args.xlsx)
    output = run / 'original_red_green'
    output.mkdir(exist_ok=True)
    labels_dir = output / 'labels'
    labels_dir.mkdir(exist_ok=True)
    truth = read_ground_truth(args.xlsx)
    with (run / 'summary.csv').open() as stream:
        tasks = list(csv.DictReader(stream))
    stats, details = [], []
    for mode in MODES:
        for case, task, base in CASES:
            group = [r for r in tasks if r['mode'] == mode and r['task'] == task]
            if not group:
                continue
            labels = np.array([truth.get(base+i, [np.nan, np.nan]) for i in range(100)])
            np.savetxt(labels_dir / f'{task}.csv', labels, delimiter=',', fmt='%.6f')
            pairs, errors = [], []
            eligible = 0
            for row in group:
                local = int(row['index'])
                gt = original.build_real_landing(labels[local])
                if gt is not None and row['status'] != 'missing_input':
                    eligible += 1
                detail = dict(mode=mode, case=case, local_id=local, excel_id=base+local,
                              run_status=row['status'], error_m='', success='')
                if gt is not None and row['status'] == 'predicted':
                    path = run / 'results' / f'{mode}_{task}' / 'ball_pixel_3D_records' / f'{local}.csv'
                    prediction = original.read_prediction_csv(path)
                    if prediction is None or not np.isfinite(prediction).all() or np.allclose(prediction, 0):
                        raise ValueError(f'Invalid prediction marked predicted: {path}')
                    error = float(np.linalg.norm(prediction[1:] - np.array(gt)[1:]))
                    pairs.append(dict(prediction=prediction, label=gt))
                    errors.append(error)
                    detail.update(error_m=error, success=error <= original.POSITION_CONFIG['success_threshold'])
                details.append(detail)
            filename = f'landing_point_simple_{mode}_{task}.png'
            original.plot_success_only(pairs, output / filename, f'{mode}_{task}')
            successes = sum(e <= original.POSITION_CONFIG['success_threshold'] for e in errors)
            stats.append(dict(method=mode, case=case, planned=len(group), green=successes,
                              red=len(pairs)-successes, valid_predictions=len(pairs),
                              success_rate_original_pct=100*successes/len(pairs) if pairs else '',
                              labeled_inputs=eligible,
                              success_rate_all_inputs_pct=100*successes/eligible if eligible else '',
                              zero_result=sum(r['status']=='no_detection' for r in group),
                              timeout=sum(r['status']=='timeout' for r in group),
                              missing_input=sum(r['status']=='missing_input' for r in group),
                              mean_error_m=float(np.mean(errors)) if errors else '', plot=filename))
    for name, rows in [('success_rates.csv', stats), ('sample_errors.csv', details)]:
        with (output / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    source = Path(original.__file__)
    manifest = dict(plotter=str(source), function='plot_success_only',
                    script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    workbook=str(args.xlsx.resolve()),
                    mapping={c:[base,base+99] for c,_,base in CASES},
                    position_config=original.POSITION_CONFIG,
                    original_denominator='Nonzero finite predictions with ground truth',
                    all_inputs_denominator='Available inputs with ground truth; zero results/timeouts are failures',
                    zero_result_note='All-zero output can mean absent observations or rejected fitting; not necessarily no ball detected.')
    (output / 'plot_config.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
