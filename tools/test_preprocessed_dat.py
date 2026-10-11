"""Processed DAT dimensions, old-file compatibility and config accessor isolation."""
from contextlib import ExitStack
from pathlib import Path
import shutil
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import config.config_loader as cfg
import config_live.config_loader as live
from dataset_tools import time_align_rewrite as preprocess
from dataset_tools import count_saved_data as counter
from data_input import read_from_aligned_buffers as replay
from tools import evaluation_batch as batch


class Packets(list):
    def get_size(self):
        return (720, 1280)


class ProcessedDatTests(unittest.TestCase):
    def events(self):
        return np.array([(0, 0, 0, 0), (1279, 0, 1, 1000),
                         (0, 719, 1, 2000), (1279, 719, 0, 3000)],
                        dtype=[('x', 'u2'), ('y', 'u2'), ('p', 'i2'), ('t', 'i8')])

    def test_raw_rotation_swaps_dimensions_and_preserves_sdk_packet(self):
        original = self.events()
        with patch.object(preprocess, '_build_events_iterator',
                          return_value=(Packets([original]), None)):
            packets, dimensions = preprocess.load_event_buffer('camera.raw', 2000, 10000)
        self.assertEqual(dimensions, (1280, 720))
        np.testing.assert_array_equal(packets[0]['x'], [0, 0, 719, 719])
        np.testing.assert_array_equal(packets[0]['y'], [1279, 0, 1279, 0])
        np.testing.assert_array_equal(original, self.events())
        np.testing.assert_array_equal(packets[0][['p', 't']], original[['p', 't']])

    def test_writer_rejects_misleading_extension(self):
        with self.assertRaisesRegex(ValueError, '.dat'):
            preprocess.save_event_buffer([self.events()], 'processed.raw')

    def test_real_sdk_roundtrip_and_legacy_raw_suffix(self):
        try:
            from metavision_core.event_io import EventsIterator
        except ImportError:
            self.skipTest('Metavision SDK not installed')
        original = self.events()
        with patch.object(preprocess, '_build_events_iterator',
                          return_value=(Packets([original]), None)):
            packets, dimensions = preprocess.load_event_buffer('camera.raw', 2000, 10000)
        with tempfile.TemporaryDirectory() as tmp:
            dat = Path(tmp) / 'processed.dat'
            self.assertTrue(preprocess.save_event_buffer(packets, dat, dimensions))
            reader = EventsIterator(input_path=str(dat), delta_t=2000)
            self.assertEqual(tuple(reader.get_size()), dimensions)
            actual = np.concatenate(list(reader))
            for field in ('x', 'y', 'p', 't'):
                np.testing.assert_array_equal(actual[field], packets[0][field])
            legacy = dat.with_suffix('.raw')
            shutil.copyfile(dat, legacy)
            # Exercise the actual SDK and legacy symlink, isolating only spatial alignment.
            with patch.object(replay.Config, 'get_alignment', return_value={}), \
                 patch.object(replay, 'align_to_rgb', side_effect=lambda ev: (ev['x'], ev['y'])):
                for path in (dat, legacy):
                    result = np.concatenate(replay.load_processed_event_file(
                        path, dict(delta_t=2000, n_events=10000)))
                    for field in ('x', 'y', 'p', 't'):
                        np.testing.assert_array_equal(result[field], packets[0][field])

    def test_dat_selection_matches_counter_and_batch_when_legacy_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'saved_data_l'
            (scene / 'event').mkdir(parents=True)
            (scene / 'rgb').mkdir()
            dat = scene / 'event/0.dat'
            raw = scene / 'event/0.raw'
            dat.touch(); raw.touch(); (scene / 'rgb/0.mp4').touch()
            self.assertEqual(Path(replay.resolve_event_path(scene, 0)), dat)
            self.assertEqual(Path(counter.resolve_event_path(scene, 0)), dat)
            item = next(p for p in batch.build_plan(root, 1) if p['task'] == 'normal')
            self.assertEqual(Path(item['event']), dat)
            dat.unlink()
            self.assertEqual(Path(replay.resolve_event_path(scene, 0)), raw)

    def test_shared_accessors_preserve_independent_stacks_and_legacy_keys(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'EVENT_RGB_DATA_ROOT': '/tmp/p2-test-data'}, clear=True))
            for module, path in ((cfg, 'config/overlays/config_l.yaml'),
                                 (live, 'config_live/overlays/config_l.yaml')):
                stack.enter_context(patch.object(module, '_config', module.ConfigLoader(path)))
            accessors = [key for key in dir(cfg.Config) if key.startswith('get_')]
            for key in accessors:
                # Supply the same stack to compare complete schemas/defaults.
                with patch.object(live, '_config', cfg._config):
                    self.assertEqual(getattr(cfg.Config, key)(), getattr(live.Config, key)())
            cfg._config.config['trajectory']['gravity'] = -3.0
            live._config.config['trajectory']['gravity'] = -7.0
            self.assertEqual(cfg.Config.get_trajectory()['gravity'], -3.0)
            self.assertEqual(live.Config.get_trajectory()['gravity'], -7.0)
            self.assertIn('impact_bounds_x', live.Config.get_trajectory())
            self.assertIn('impact_bounds_y', live.Config.get_trajectory())
            self.assertEqual(cfg.Config.get_camera_intrinsics()['dist_coeffs'],
                             cfg._config.get('camera_intrinsics.distortion_coeffs'))


if __name__ == '__main__':
    unittest.main()
