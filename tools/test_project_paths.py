"""Portable path resolution and entrypoint wiring without cameras/GPU inference."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import project_paths as paths
from config.config_loader import ConfigLoader
from dataset_tools import collect_data as capture
from dataset_tools import time_align_rewrite as preprocess
from dataset_tools import prepare_dataset as prepare
from dataset_tools import count_saved_data as counter

PROJECT = Path(__file__).resolve().parents[1]


def capture_save_child(save_dir, result):
    # Run through the production worker wrapper under spawn. Do not inherit globals.
    from dataset_tools import collect_data
    writer = Mock()
    writer.isOpened.return_value = True
    with patch.object(collect_data.cv2, 'VideoWriter', return_value=writer) as factory:
        collect_data.save_rgb_frames(7, [np.zeros((1, 1, 3), dtype=np.uint8)], save_dir)
    result.put(factory.call_args[0][0])


class PathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.addCleanup(patch.stopall)
        patch.object(paths, 'PROJECT_ROOT', self.root).start()
        patch.dict(os.environ, {}, clear=True).start()
        (self.root / 'config').mkdir()
        self.local = self.root / 'config/paths.local.yaml'
        self.local.write_text('data_root: machine-a\n')

    def test_machine_root_env_and_explicit_paths_have_correct_priority(self):
        self.assertEqual(paths.stage_path('raw', 'dark'), self.root / 'machine-a/saved_data_raw/saved_data_d')
        os.environ['EVENT_RGB_DATA_ROOT'] = str(self.root / 'machine-b')
        self.assertEqual(paths.stage_path('dataset', 'dark'), self.root / 'machine-b/saved_data/dataset_d')
        os.environ['CAPTURE_DIR'] = 'environment/raw'
        self.assertEqual(paths.stage_path('raw'), self.root / 'environment/raw')
        self.assertEqual(paths.stage_path('raw', explicit='explicit/raw'), self.root / 'explicit/raw')
        os.environ['DATA_ROOT'] = 'legacy-processed'
        self.assertEqual(paths.stage_path('replay', 'clutter'), self.root / 'legacy-processed/saved_data_c')
        os.environ['RECORD_DIR'] = 'exact-recording'
        self.assertEqual(paths.stage_path('replay'), self.root / 'exact-recording')
        self.assertEqual(paths.stage_path('replay', explicit='cli'), self.root / 'cli')

    def test_relative_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir('/tmp')
            self.assertEqual(paths.resolve_path('input'), self.root / 'input')
            self.assertEqual(paths.resolve_path('${data_root}/processed_data'), self.root / 'machine-a/processed_data')
        finally:
            os.chdir(previous)

    def test_missing_or_invalid_local_setting_does_not_guess_a_directory(self):
        self.local.unlink()
        with self.assertRaisesRegex(ValueError, 'EVENT_RGB_DATA_ROOT'):
            paths.stage_path('raw')
        self.assertEqual(paths.stage_path('raw', explicit='explicit'), self.root / 'explicit')
        self.local.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'mapping'):
            paths.data_root()
        self.local.write_text('data_root: ""')
        with self.assertRaisesRegex(ValueError, 'empty'):
            paths.data_root()

    def test_both_config_stacks_resolve_all_scenes_and_resource_paths(self):
        for stack in ('config', 'config_live'):
            for overlay in (PROJECT / stack / 'overlays').glob('*.yaml'):
                config = ConfigLoader(overlay)
                resolved = paths.config_paths(config)
                self.assertTrue(resolved['record_dir'].startswith(str(self.root / 'machine-a/processed_data/')))
                self.assertEqual(resolved['evaluation_root'], str(self.root / 'machine-a/test_data/evaluation_results'))
                self.assertTrue(resolved['template_npy'].startswith(str(self.root / 'tracking/')))
        os.environ.update(RECORD_DIR='custom/record', EVALUATION_RESULTS_DIR='custom/results')
        resolved = paths.config_paths(config)
        self.assertEqual(resolved['record_dir'], str(self.root / 'custom/record'))
        self.assertEqual(resolved['evaluation_root'], str(self.root / 'custom/results'))

    def test_stage_entrypoints_agree_on_raw_processed_and_training_directories(self):
        for task in paths.SCENES:
            recorded = capture.parse_args(['--task', task]).save_dir
            aligned = preprocess.parse_args(['--task', task])
            training = prepare.parse_args(['--task', task])
            counted = counter.parse_args(['--task', task])
            self.assertEqual(str(recorded), aligned.record_dir)
            self.assertEqual(training.record_dir, str(recorded))
            self.assertEqual(aligned.save_dir, counted.record_dir)
            self.assertNotEqual(training.output_dir, aligned.save_dir)
            self.assertEqual(training.index, 13)
            self.assertEqual(training.count, 87)

    def test_explicit_stage_paths_work_without_a_machine_config(self):
        self.local.unlink()
        self.assertEqual(capture.parse_args(['--save-dir', 'raw']).save_dir, self.root / 'raw')
        args = preprocess.parse_args(['--record-dir', 'raw', '--save-dir', 'processed'])
        self.assertEqual(args.record_dir, str(self.root / 'raw'))
        self.assertEqual(args.save_dir, str(self.root / 'processed'))
        args = prepare.parse_args(['--record-dir', 'raw', '--output-dir', 'training'])
        self.assertEqual(args.output_dir, str(self.root / 'training'))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            preprocess.parse_args(['--record-dir', 'same', '--save-dir', 'same'])

    def test_missing_input_fails_before_processing(self):
        with patch.object(sys, 'argv', ['preprocess', '--record-dir', 'missing', '--save-dir', 'output']), \
                patch.object(preprocess, 'run_index') as run:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing input directory'):
                preprocess.main()
            run.assert_not_called()
        with patch.object(sys, 'argv', ['prepare', '--record-dir', 'missing', '--output-dir', 'training']), \
                patch.object(prepare, 'align_buffer') as run:
            with self.assertRaises(FileNotFoundError):
                prepare.main()
            run.assert_not_called()
        self.assertFalse((self.root / 'output').exists())

    def test_historical_paths_are_explicit_and_validated_before_importing_old_code(self):
        from evaluation_tools.plot_old_model_new_gt import legacy_options
        source = self.root / 'old-source'
        (source / 'fitting').mkdir(parents=True)
        (source / 'fitting/parabola_fitter.py').touch()
        (source / 'utils.py').touch()
        tracks = self.root / 'old-tracks'
        tracks.mkdir()
        with redirect_stdout(io.StringIO()):
            args = legacy_options('plots', ['--legacy-source-root', 'old-source', '--tracks-dir', 'old-tracks'])
        self.assertEqual(args.legacy_source_root, source)
        self.assertEqual(args.tracks_dir, tracks)
        self.assertEqual(args.output_dir, self.root / 'plots')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            legacy_options('plots', ['--legacy-source-root', 'missing', '--tracks-dir', 'old-tracks'])

    def test_tool_help_works_from_other_directory_without_hardware_or_local_config(self):
        files = ['dataset_tools/collect_data.py', 'dataset_tools/time_align_rewrite.py',
                 'dataset_tools/prepare_dataset.py', 'dataset_tools/count_saved_data.py',
                 'dataset_tools/pick_positive.py', 'dataset_tools/dataset_viewer.py',
                 'dataset_tools/update_absent_by_flags.py', 'dataset_tools/adjust_groundtruth_y.py',
                 'evaluation_tools/plot_old_model_new_gt.py', 'evaluation_tools/run_legacy_plots.py',
                 'evaluation_tools/plot_landing_prediction_simple.py',
                 'evaluation_tools/plot_landing_prediction_multi.py',
                 'evaluation_tools/plot_landing_error_boxplot.py',
                 'evaluation_tools/plot_replay_vs_excel.py', 'evaluation_tools/analyze_ball_csvs.py']
        for name in files:
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, str(PROJECT / name), '--help'],
                                        cwd=self.root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_capture_reserves_index_in_selected_folder_and_passes_it_to_workers(self):
        target = self.root / 'recorded'
        for folder, ext, index in (('event', 'raw', 7), ('rgb', 'mp4', 9)):
            (target / folder).mkdir(parents=True)
            (target / folder / f'{index}.{ext}').touch()
        self.assertEqual(capture.next_record_index(target), 10)
        context = multiprocessing.get_context('spawn')
        result = context.Queue()
        worker = context.Process(target=capture.run_worker,
                                 args=(capture_save_child, (target, result)))
        worker.start()
        try:
            self.assertEqual(result.get(timeout=10), str(target / 'rgb/7.mp4'))
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join(3)
            result.close()
            result.join_thread()
        # Check production parent wiring: both camera workers receive the resolved path.
        tree = ast.parse((PROJECT / 'dataset_tools/collect_data.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'workers' for t in n.targets))
        ns = dict(save_dir=target, record_flir=Mock(), record_event=Mock(), keyboard_listener=Mock(),
                  show_streams=Mock(), shared_flir_meta=object(), shared_event_meta=object(), session=object())
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<workers>', 'exec'), ns)
        for name, _, args in ns['workers'][:2]:
            self.assertEqual(args[-1], target, name)

    def test_shell_root_override_beats_inherited_record_dir(self):
        data = self.root / 'processed'
        for kind, ext in (('event', 'dat'), ('rgb', 'mp4')):
            (data / 'saved_data_l' / kind).mkdir(parents=True)
            (data / 'saved_data_l' / kind / f'0.{ext}').touch()
        env = dict(os.environ, RECORD_DIR='/wrong/inherited', DATA_DIR='/wrong/inherited',
                   ACTIVE_CONFIG_FILE=str(PROJECT / 'config/overlays/config_l_event.yaml'))
        result = subprocess.run(['bash', str(PROJECT / 'run_record.sh'), '--data-root', str(data), '--dry-run'],
                                cwd='/tmp', env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(data / 'saved_data_l'), result.stdout)
        # Shared total-root env also selects processed_data when no legacy override exists.
        env.pop('RECORD_DIR')
        env.pop('DATA_DIR')
        env['EVENT_RGB_DATA_ROOT'] = str(self.root)
        # Explicit stage folder name, with no fallback to an existing sibling.
        result = subprocess.run(['bash', str(PROJECT / 'run_record.sh'), '--dry-run'],
                                cwd='/tmp', env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(str(self.root / 'processed_data/saved_data_l/event'), result.stderr)


if __name__ == '__main__':
    unittest.main()
