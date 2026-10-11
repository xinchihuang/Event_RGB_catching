import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluation_batch as batch
import config.config_loader as config_module


class BatchTests(unittest.TestCase):
    def test_plan_keeps_ids_and_excludes_extra_dark_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case in ['l', 'c', 'd', 'dc']:
                folder = root / f'saved_data_{case}'
                for sub, ext in [('event', 'raw'), ('rgb', 'mp4')]:
                    (folder / sub).mkdir(parents=True)
                    for i in range(101):
                        if case == 'l' and i == 12:
                            continue
                        (folder / sub / f'{i}.{ext}').touch()
            plan = batch.build_plan(root)
            self.assertEqual(len(plan), 800)
            self.assertEqual(sum(p['missing_input'] for p in plan), 3)
            self.assertTrue(all(p['index'] < 100 for p in plan))
            self.assertEqual(next(p['excel_id'] for p in plan if p['case'] == 'dc'), 200)
            self.assertEqual(next(p['excel_id'] for p in plan if p['case'] == 'd'), 300)
            self.assertEqual({(p['mode'], p['task']) for p in plan},
                             {('Model', 'normal'), ('Model', 'clutter'), ('Model', 'dark'),
                              ('Model', 'dark_clutter'), ('RGB', 'normal'), ('RGB', 'clutter'),
                              ('Event', 'normal'), ('Event', 'dark')})

    def test_prediction_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / '0.csv'
            self.assertEqual(batch.prediction_status(p), 'missing_prediction')
            for value, expected in [('0,0,0', 'no_detection'), ('0,.2,1.2', 'predicted'),
                                    ('0,nan,1', 'invalid_prediction'), ('', 'invalid_prediction')]:
                p.write_text(value)
                self.assertEqual(batch.prediction_status(p), expected)

    def test_process_exit_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'worker.log'
            result = batch.run_sample([sys.executable, '-c', 'raise SystemExit(7)'],
                                      os.environ.copy(), path, 5, lambda: None)
            self.assertEqual(result, ('process_failed', 7))
            result = batch.run_sample([sys.executable, '-c', 'import time; time.sleep(60)'],
                                      os.environ.copy(), path, .1, lambda: None)
            self.assertEqual(result, ('timeout', 124))

    def test_batch_continues_after_missing_and_failed_sample(self):
        import csv
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'evaluation_tools').mkdir()
            (root / 'config/overlays').mkdir(parents=True)
            (root / 'config/camera_config.yaml').write_text('alignment: {replay_apply_rotation: true}')
            (root / 'config/tracking_config.yaml').write_text('{}')
            data = root / 'data'
            data.mkdir()
            for mode, task, case in batch.JOBS:
                (root / f'config/overlays/config_{case}_{mode.lower()}.yaml').write_text('paths: {}')
                if case != 'l':
                    for sub, ext in [('event', 'raw'), ('rgb', 'mp4')]:
                        folder = data / f'saved_data_{case}' / sub
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / f'0.{ext}').touch()
            output = root / 'output'
            args = ['batch', '--data-root', str(data), '--output-root', str(output), '--limit', '1',
                    '--ratio', '3']
            with patch.object(batch, 'PROJECT', root), patch.object(config_module, 'PROJECT_ROOT', root), \
                 patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', args), \
                 patch.dict(sys.modules, {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))}), \
                 patch.object(batch, 'run_sample', return_value=('process_failed', 1)) as run, \
                 patch.object(batch.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.main(), 2)
            with (output / 'summary.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(run.call_count, 5)
            for call in run.call_args_list:
                command = call[0][0]
                self.assertEqual(command[command.index('--ratio') + 1], '3')
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['state'], 'completed_with_errors')
            self.assertEqual(status['processed'], 8)
            self.assertEqual(status['ratio'], 3)
            self.assertEqual(len(status['completed_cases']), 8)
            child_env = run.call_args[0][1]
            import yaml
            for path in (output / 'configs').glob('*.yaml'):
                self.assertIs(yaml.safe_load(path.read_text())['alignment']['replay_apply_rotation'], False)
            self.assertEqual(child_env['ENABLE_GUI'], '0')
            self.assertEqual(child_env['EVALUATION_RESULTS_DIR'], str(output / 'results'))
            self.assertEqual(status['counts'], {'missing_input': 3, 'process_failed': 5})


if __name__ == '__main__':
    unittest.main()
