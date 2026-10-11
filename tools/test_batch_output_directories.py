"""Batch outputs must stay together without overwriting earlier invocations."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_record


class BatchOutputTests(unittest.TestCase):
    def test_multiple_samples_share_one_run_and_repeated_batches_are_isolated(self):
        configs = [
            dict(test_mode='Event', task='normal', start_index=0, max_runs=2, restart_delay=0),
            dict(test_mode='RGB', task='clutter', start_index=0, max_runs=2, restart_delay=0),
        ]
        calls = []

        def replay(command, *, env, **kwargs):
            # Simulate record.py's real result path contract, without inference.
            index = command[command.index('--record_index') + 1]
            mode = command[command.index('--test_mode') + 1]
            task = command[command.index('--task') + 1]
            root = Path(env['EVALUATION_RESULTS_DIR'])
            for folder in ('ball_pixel_3D_records', 'ball_pixel_3D_records_ball'):
                path = root / f'{mode}_{task}' / folder / f'{index}.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'call={len(calls)}\n')
            calls.append(root)
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            old_results = str(project / 'historical-results')
            with patch.object(run_record, 'PROJECT', project), \
                 patch.object(run_record.signal, 'signal'), \
                 patch.object(run_record.subprocess, 'run', side_effect=replay), \
                 patch.dict(sys.modules, {'cv2': types.SimpleNamespace(destroyAllWindows=lambda: None,
                                                                      waitKey=lambda _: None)}), \
                 patch.dict(os.environ, {'EVALUATION_RESULTS_DIR': old_results}), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = run_record.run_batch_config(configs)
                first_content = {p.relative_to(first): p.read_text() for p in first.rglob('*.csv')}
                second = run_record.run_batch_config(configs)
                self.assertEqual(os.environ['EVALUATION_RESULTS_DIR'], old_results)

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, project / 'evaluation_tools/batch_runs')
            self.assertEqual(calls[:4], [first / 'results'] * 4)
            self.assertEqual(calls[4:], [second / 'results'] * 4)
            self.assertEqual(len(first_content), 8)
            self.assertEqual(first_content,
                             {p.relative_to(first): p.read_text() for p in first.rglob('*.csv')})
            self.assertEqual(json.loads((first / 'batch_config.json').read_text()),
                             [dict(config, ratio=10) for config in configs])
            self.assertFalse(Path(old_results).exists())

    def test_explicit_new_directory_is_used_and_existing_one_is_rejected(self):
        config = dict(test_mode='Event', task='normal', max_runs=1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'chosen batch'
            with patch.object(run_record.signal, 'signal'), \
                 patch.object(run_record.RecordRunner, 'run_loop') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_record.run_batch_config(config, output), output)
                snapshot = (output / 'batch_config.json').read_text()
                with self.assertRaises(FileExistsError):
                    run_record.run_batch_config(config, output)
                run.assert_called_once()
            self.assertEqual((output / 'batch_config.json').read_text(), snapshot)

    def test_cli_passes_output_directory_to_batch(self):
        with patch.object(sys, 'argv', ['run_record.py', '--batch_config', '{"max_runs": 1}',
                                       '--output-root', 'new batch']), \
             patch.object(run_record, 'run_batch_config') as run:
            run_record.main()
        run.assert_called_once_with({'max_runs': 1}, output_root=Path('new batch'), ratio=None)


if __name__ == '__main__':
    unittest.main()
