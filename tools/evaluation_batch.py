"""Finite offline evaluation: existing record.py pipeline, eight combinations."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from config.config_loader import ConfigLoader
from replay_options import positive_ratio
from project_paths import stage_path, resolve_path
JOBS = [('Model', 'normal', 'l'), ('Model', 'clutter', 'c'),
        ('Model', 'dark', 'd'), ('Model', 'dark_clutter', 'dc'),
        ('RGB', 'normal', 'l'), ('RGB', 'clutter', 'c'),
        ('Event', 'normal', 'l'), ('Event', 'dark', 'd')]
OFFSETS = dict(l=0, c=100, dc=200, d=300)


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)


def data_directory(root, case):
    return root / f'saved_data_{case}'


def build_plan(root, limit=None):
    plan = []
    for mode, task, case in JOBS:
        directory = data_directory(root, case)
        for index in range(100 if limit is None else limit):
            event = next((directory / 'event' / f'{index}.{ext}' for ext in ['dat', 'raw']
                          if (directory / 'event' / f'{index}.{ext}').is_file()), None)
            video = directory / 'rgb' / f'{index}.mp4'
            plan.append(dict(mode=mode, task=task, case=case, index=index,
                             excel_id=OFFSETS[case]+index, record_dir=str(directory),
                             event=str(event) if event else None, video=str(video),
                             missing_input=event is None or not video.is_file()))
    return plan


def stop_group(process):
    # Children share this session; stop them even if the parent has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_sample(command, env, log_path, timeout, heartbeat):
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        started = time.monotonic()
        try:
            while process.poll() is None:
                heartbeat()
                if time.monotonic() - started >= timeout:
                    return 'timeout', 124
                time.sleep(.5)
            return ('exited' if process.returncode == 0 else 'process_failed'), process.returncode
        finally:
            stop_group(process)


def prediction_status(path):
    if not path.is_file():
        return 'missing_prediction'
    try:
        with path.open() as handle:
            rows = list(csv.reader(handle))
        if len(rows) != 1 or len(rows[0]) != 3:
            return 'invalid_prediction'
        values = [float(v) for v in rows[0]]
        if not all(math.isfinite(v) for v in values):
            return 'invalid_prediction'
        return 'no_detection' if all(v == 0 for v in values) else 'predicted'
    except (ValueError, OSError):
        return 'invalid_prediction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Already rotated processed recordings.',
                        default=None)
    parser.add_argument('--output-root', type=Path, help='New directory; existing runs are never overwritten.')
    parser.add_argument('--ratio', type=positive_ratio, default=10,
                        help='Event packets per RGB frame (default: 10).')
    parser.add_argument('--timeout', type=float, default=180, help='Seconds per sample.')
    parser.add_argument('--limit', type=int, help='For smoke tests: first N IDs per combination (1–100).')
    parser.add_argument('--gui', dest='gui', action='store_true', help='Show replay windows; close after each sample.')
    parser.add_argument('--no-gui', dest='gui', action='store_false')
    parser.add_argument('--plot-xlsx', type=Path, help='Generate original red/green plots and success rates after replay.')
    parser.set_defaults(gui=False)
    parser.add_argument('--dry-run', action='store_true', help='Check input coverage without requiring CUDA.')
    args = parser.parse_args()
    if args.timeout <= 0 or (args.limit is not None and not 1 <= args.limit <= 100):
        parser.error('timeout must be positive; limit must be 1–100')
    source_configs = {}
    try:
        for mode, task, case in JOBS:
            requested = (os.environ["ACTIVE_CONFIG_FILE"] if "ACTIVE_CONFIG_FILE" in os.environ
                         else PROJECT / f"config/overlays/config_{case}_{mode.lower()}.yaml")
            source_configs[(mode, task)] = ConfigLoader(requested).config
    except (OSError, ValueError) as exc:
        parser.error(f"Configuration error: {exc}")
    try:
        args.data_root = stage_path('processed', explicit=args.data_root)
        if args.output_root is not None:
            args.output_root = resolve_path(args.output_root)
        if args.plot_xlsx is not None:
            args.plot_xlsx = resolve_path(args.plot_xlsx)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Processed data root: {args.data_root}", flush=True)
    if not args.data_root.is_dir():
        parser.error(f'Data root is unavailable: {args.data_root}')
    if args.plot_xlsx and not args.plot_xlsx.is_file():
        parser.error(f'Ground truth workbook is unavailable: {args.plot_xlsx}')
    plan = build_plan(args.data_root.resolve(), args.limit)
    for mode, task, _ in JOBS:
        items = [p for p in plan if p['mode'] == mode and p['task'] == task]
        print(f'{mode}_{task}: {sum(not p["missing_input"] for p in items)} available, '
              f'{sum(p["missing_input"] for p in items)} missing', flush=True)
    print(f'Total planned: {len(plan)} method/sample pairs.', flush=True)
    if args.dry_run:
        return 0
    import torch
    if not torch.cuda.is_available():
        parser.error('Model requires CUDA. Run this launcher in the terminal where torch.cuda.is_available() is True.')
    if not any(not p['missing_input'] for p in plan):
        parser.error('No paired RAW/DAT and MP4 recordings found.')

    runs = PROJECT / 'evaluation_tools/batch_runs'
    runs.mkdir(exist_ok=True)
    lock = (runs / '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another evaluation batch is already running.')
    output = (args.output_root or runs / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ['logs', 'configs', 'results']:
        (output / folder).mkdir()
    atomic_json(output / 'plan.json', plan)
    atomic_json(runs / 'latest.json', dict(output_root=str(output)))
    state = dict(state='running', pid=os.getpid(), started_at=datetime.now().astimezone().isoformat(),
                 total=len(plan), processed=0, counts={}, current=None, completed_cases=[],
                 gui=args.gui, timeout=args.timeout, ratio=args.ratio, output_root=str(output))
    rows = []

    def heartbeat():
        state['heartbeat_at'] = datetime.now().astimezone().isoformat()
        atomic_json(output / 'status.json', state)

    def on_signal(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    heartbeat()
    print(f'Output and status: {output}', flush=True)
    configs = {}
    try:
        for mode, task, case in JOBS:
            # Save the validated merged stack, including explicit live/external settings.
            config = source_configs[(mode, task)]
            config.setdefault('paths', {}).update(
                record_dir=str(data_directory(args.data_root.resolve(), case)),
                evaluation_root=str(output / 'results'))
            # All replay inputs are already rotated; never inherit legacy raw settings.
            config.setdefault('alignment', {})['replay_apply_rotation'] = False
            path = output / 'configs' / f'{mode}_{task}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            configs[(mode, task)] = path
        with (output / 'summary.csv').open('w', newline='') as summary:
            fields = ['mode', 'task', 'index', 'excel_id', 'status', 'returncode', 'seconds', 'log', 'prediction']
            writer = csv.DictWriter(summary, fieldnames=fields)
            writer.writeheader()
            for item in plan:
                mode, task, index = item['mode'], item['task'], item['index']
                state['current'] = dict(mode=mode, task=task, index=index)
                heartbeat()
                case_key = f'{mode}_{task}'
                if index == 0:
                    print(f'=== Starting case {case_key} ===', flush=True)
                row = {k: item[k] for k in ['mode', 'task', 'index', 'excel_id']}
                started = time.monotonic()
                if item['missing_input']:
                    row['status'] = 'missing_input'
                else:
                    env = os.environ.copy()
                    env.pop('PYTHONNOUSERSITE', None)
                    env.update(ENABLE_GUI='1' if args.gui else '0', ACTIVE_CONFIG_FILE=str(configs[(mode, task)]),
                               ACTIVE_TASK=task, TEST_MODE=mode, PYTHONUNBUFFERED='1',
                               RECORD_DIR=item['record_dir'], EVALUATION_RESULTS_DIR=str(output / 'results'))
                    log = output / 'logs' / f'{mode}_{task}_{index:03d}.log'
                    prediction = output / 'results' / f'{mode}_{task}' / 'ball_pixel_3D_records' / f'{index}.csv'
                    row.update(log=str(log), prediction=str(prediction))
                    result, code = run_sample(
                        [sys.executable, '-u', str(PROJECT / 'record.py'), '--test_mode', mode,
                         '--task', task, '--record_index', str(index), '--ratio', str(args.ratio)],
                        env, log, args.timeout, heartbeat)
                    row.update(status=prediction_status(prediction) if result == 'exited' else result,
                               returncode=code)
                row['seconds'] = round(time.monotonic()-started, 2)
                writer.writerow(row)
                summary.flush()
                rows.append(row)
                state.update(processed=len(rows), counts=dict(Counter(r['status'] for r in rows)))
                heartbeat()
                print(f'[{len(rows)}/{len(plan)}] {mode}_{task}/{index}: {row["status"]}', flush=True)
                if index == (args.limit or 100) - 1:
                    case_counts = dict(Counter(r['status'] for r in rows if r['mode'] == mode and r['task'] == task))
                    state['completed_cases'].append(dict(case=case_key, counts=case_counts))
                    heartbeat()
                    print(f'=== Finished case {case_key}: {case_counts}; processes/windows closed ===', flush=True)
        errors = {'timeout', 'process_failed', 'invalid_prediction', 'missing_prediction'}
        state['state'] = 'completed_with_errors' if any(r['status'] in errors for r in rows) else 'completed'
        if args.plot_xlsx:
            state['plot_state'] = 'running'
            heartbeat()
            with (output / 'plot.log').open('w') as log:
                result = subprocess.run(
                    [sys.executable, str(PROJECT / 'tools/plot_batch_red_green.py'),
                     '--run-dir', str(output), '--xlsx', str(args.plot_xlsx.resolve())],
                    cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            state['plot_state'] = 'completed' if result.returncode == 0 else 'failed'
            if result.returncode:
                state['state'] = 'completed_with_errors'
    except KeyboardInterrupt:
        state['state'] = 'interrupted'
    except Exception as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['ended_at'] = datetime.now().astimezone().isoformat()
        heartbeat()
        lock.close()
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return {'completed': 0, 'completed_with_errors': 2, 'interrupted': 130}.get(state['state'], 1)


if __name__ == '__main__':
    sys.exit(main())
