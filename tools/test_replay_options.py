"""File-replay ratio and counting CLI checks without SDKs or recorded data."""
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from data_input import read_from_aligned_buffers as replay
from dataset_tools import count_saved_data as counter
from replay_options import positive_ratio
import run_record


PROJECT = Path(__file__).resolve().parents[1]


class RatioTests(unittest.TestCase):
    def test_validation_rejects_invalid_values_before_shared_memory(self):
        for value in (0, -1, 1.5, 2.0, True, None, '0', '2.5', ''):
            with self.subTest(value=value), patch.object(replay.shared_memory, 'SharedMemory') as shm:
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    replay.read_from_aligned_buffers(*([None] * 12), ratio=value)
                shm.assert_not_called()
        self.assertEqual(positive_ratio('3'), 3)
        self.assertEqual(positive_ratio(np.int64(4)), 4)

    def test_ratio_controls_actual_rgb_updates_and_trimming(self):
        for ratio, event_total in ((2, 10), (3, 12), (10, 12)):
            with self.subTest(ratio=ratio):
                state = SimpleNamespace(value='w')
                event_flag, rgb_flag = SimpleNamespace(value='s'), SimpleNamespace(value='s')
                event_frame, rgb_frame = SimpleNamespace(value=-1), SimpleNamespace(value=-1)
                memories = {name: SimpleNamespace(name=name, buf=bytearray(8))
                            for name in ('events', 'rgb')}
                rgb_view = np.ndarray((1,), dtype=np.int64, buffer=memories['rgb'].buf)
                trace = []
                iterations = 0

                def stopped():
                    nonlocal iterations
                    iterations += 1
                    if iterations > 50:
                        raise AssertionError('Replay did not finish')
                    if event_flag.value == 'f':
                        return True
                    if event_frame.value >= 0:
                        trace.append((event_frame.value, rgb_frame.value, int(rgb_view[0])))
                    return False

                def rgb_loader(_):
                    state.value = 't'
                    return [np.array([100 + i], dtype=np.int64) for i in range(5)]

                values = [SimpleNamespace(value=0) for _ in range(6)]
                event_version, rgb_version, num_event, raw_count, index, start_ts = values
                with patch.object(replay.shared_memory, 'SharedMemory', side_effect=lambda name: memories[name]), \
                        patch.object(replay, '_load_event_buffer', return_value=[np.array([i]) for i in range(12)]), \
                        patch.object(replay, '_load_rgb_buffer', side_effect=rgb_loader), \
                        patch.object(replay.Config, 'get_data_reading', return_value={'send_interval': 0}), \
                        patch.object(replay.Config, 'get_image_processing', return_value={'brightness_multiplier': 1}), \
                        redirect_stdout(io.StringIO()):
                    kwargs = {} if ratio == 10 else {'ratio': ratio}
                    replay.read_from_aligned_buffers(
                        (memories['events'], np.int64, (1,)), (memories['rgb'], np.int64, (1,)),
                        event_version, rgb_version, state, num_event, raw_count, index, start_ts,
                        SimpleNamespace(is_set=stopped), event_flag, rgb_flag,
                        current_event_frame=event_frame, current_rgb_frame=rgb_frame, **kwargs)
                self.assertEqual(raw_count.value, event_total)
                self.assertEqual(trace, [(i, i // ratio, 100 + i // ratio) for i in range(event_total)])
                self.assertEqual(rgb_flag.value, 'f')

    def startup(self, filename, argv):
        tree = ast.parse((PROJECT / filename).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.If)
                    and any(isinstance(x, ast.Name) and x.id == '__name__' for x in ast.walk(n.test)))
        end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
        namespace = dict(argparse=argparse)
        with patch.object(sys, 'argv', [filename] + argv):
            exec(compile(ast.Module(body=main.body[:end + 1], type_ignores=[]), filename, 'exec'), namespace)
        return tree, main, namespace

    def test_replay_entrypoint_parsers_and_worker_arguments(self):
        for filename in ('record.py', 'record_compare.py'):
            for argv, expected in (([], 10), (['--ratio', '3'], 3)):
                with self.subTest(filename=filename, argv=argv):
                    tree, _, namespace = self.startup(filename, argv)
                    self.assertEqual(namespace['args'].ratio, expected)
                    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                                and isinstance(n.func, ast.Name) and n.func.id == 'Process'
                                and any(k.arg == 'target' and isinstance(k.value, ast.Name)
                                        and k.value.id == 'read_from_aligned_buffers' for k in n.keywords))
                    ns = {n.id: object() for n in ast.walk(call) if isinstance(n, ast.Name)}
                    ns.update(args=namespace['args'], Process=Mock())
                    eval(compile(ast.Expression(call), filename, 'eval'), ns)
                    self.assertEqual(ns['Process'].call_args[1]['kwargs']['ratio'], expected)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                self.startup(filename, ['--ratio', '0'])
            self.assertEqual(exc.exception.code, 2)

    def test_comparison_all_and_save_forward_ratio_to_every_child(self):
        for mode, child_count in (('All', 4), ('Save', 3)):
            tree, main, namespace = self.startup('record_compare.py', ['--test_mode', mode, '--ratio', '4'])
            branch = next(n for n in main.body if isinstance(n, ast.If)
                          and any(isinstance(x, ast.Constant) and x.value == 'save' for x in ast.walk(n.test)))
            run = Mock(return_value=SimpleNamespace(returncode=0))
            namespace.update(sys=sys, os=os, subprocess=SimpleNamespace(run=run),
                             __file__=str(PROJECT / 'record_compare.py'))
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exc:
                exec(compile(ast.Module(body=[branch], type_ignores=[]), 'record_compare.py', 'exec'), namespace)
            self.assertEqual(exc.exception.code, 0)
            self.assertEqual(run.call_count, child_count)
            for call in run.call_args_list:
                command = call[0][0]
                self.assertEqual(command[command.index('--ratio') + 1], '4')

    def test_old_batch_json_ratio_and_cli_override(self):
        for override, expected in ((None, [3, 10]), (5, [5, 5])):
            with tempfile.TemporaryDirectory() as tmp, patch.object(run_record.signal, 'signal'), \
                    patch.object(run_record.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                    patch.object(run_record.RecordRunner, 'run_loop', new=lambda self: self.run_record_process()), \
                    redirect_stdout(io.StringIO()):
                config = [{'ratio': 3}, {}]
                output = run_record.run_batch_config(config, Path(tmp) / 'batch', ratio=override)
                self.assertEqual(config, [{'ratio': 3}, {}])
                self.assertEqual([row['ratio'] for row in json.loads((output / 'batch_config.json').read_text())], expected)
                self.assertEqual([int(call[0][0][call[0][0].index('--ratio') + 1]) for call in run.call_args_list], expected)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'invalid'
            with self.assertRaises(ValueError):
                run_record.run_batch_config([{'ratio': 0}], output)
            self.assertFalse(output.exists())

    def test_shell_forwards_ratio_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, suffix in (('event', 'dat'), ('rgb', 'mp4')):
                (root / folder).mkdir()
                (root / folder / f'0.{suffix}').touch()
            env = dict(os.environ, TEST_MODE='Event', TASK='normal', RECORD_INDEX='0',
                       ACTIVE_CONFIG_FILE=str(PROJECT / 'config/overlays/config_l_event.yaml'))
            for option in (['--ratio', '3'], ['--ratio=3']):
                result = subprocess.run(['bash', str(PROJECT / 'run_record.sh'), '--data-dir', tmp,
                                         '--dry-run'] + option, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--ratio 3', result.stdout)
            for value in ('0', '-1', '2.5', ''):
                result = subprocess.run(['bash', str(PROJECT / 'run_record.sh'), '--ratio=' + value],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)


class CountTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'EVENT_RGB_DATA_ROOT': '/tmp/count-test-data'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_selection_defaults_and_conflicts(self):
        args = counter.parse_args([])
        self.assertEqual((args.index, args.count, args.auto, args.all), (0, 1, False, False))
        for argv in (['--all', '--auto'], ['--all', '--index', '0'],
                     ['--all', '--count', '2'], ['--auto', '--count', '2'],
                     ['--count', '0'], ['--index', '-1']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                counter.parse_args(argv)
            self.assertEqual(exc.exception.code, 2)

    def test_count_ranges_auto_gaps_and_all_sum_actual_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, suffix, indices in (('event', 'dat', [0, 1, 3]), ('rgb', 'mp4', [0, 1, 2, 3])):
                (root / folder).mkdir()
                for i in indices:
                    (root / folder / f'{i}.{suffix}').touch()
            cases = [([], [0]), (['--index', '1'], [1]),
                     (['--count', '4'], [0, 1, 3]), (['--auto'], [0, 1]),
                     (['--auto', '--index', '3'], [3]), (['--all'], [0, 1, 3])]
            for options, expected in cases:
                with self.subTest(options=options), patch.object(sys, 'argv',
                        ['count', '--record-dir', tmp] + options), \
                        patch.object(counter, 'count_event_packets', return_value=20) as count_events, \
                        patch.object(counter, 'count_rgb_frames', return_value=2), \
                        redirect_stdout(io.StringIO()) as output:
                    counter.main()
                self.assertEqual([int(Path(call[0][0]).stem) for call in count_events.call_args_list], expected)
                if len(expected) > 1:
                    self.assertIn(f'total rgb frames = {2 * len(expected)}', output.getvalue())
                    self.assertIn(f'total event packets = {20 * len(expected)}', output.getvalue())


if __name__ == '__main__':
    unittest.main()
