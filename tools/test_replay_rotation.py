import unittest
from unittest.mock import Mock, patch
import numpy as np
from data_input import read_from_aligned_buffers as replay


class Packets(list):
    def get_size(self):
        # Old processed DAT headers may still report the original sensor size.
        return (720, 1280)


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.events = np.array(
            [(0, 0, 1, 10), (719, 0, 0, 20), (0, 1279, 1, 30), (719, 1279, 0, 40)],
            dtype=[('x', 'u2'), ('y', 'u2'), ('p', 'i2'), ('t', 'i8')],
        )

    def test_processed_coordinates_preserved_and_rgb_alignment_applied(self):
        original = self.events.copy()
        cleanup = Mock()
        with patch.object(replay, '_build_events_iterator',
                          return_value=(Packets([self.events]), cleanup)), \
             patch.object(replay.Config, 'get_alignment',
                          return_value=dict(apply_rotation=True, replay_apply_rotation=False)), \
             patch.object(replay, 'align_to_rgb',
                          side_effect=lambda ev: (ev['x'] + 10, ev['y'] + 20)) as align:
            result = replay.load_processed_event_file('processed.raw', dict(delta_t=2000, n_events=10000))[0]
        align.assert_called_once()
        cleanup.assert_called_once()
        np.testing.assert_array_equal(result['x'], original['x'] + 10)
        np.testing.assert_array_equal(result['y'], original['y'] + 20)
        np.testing.assert_array_equal(result['p'], original['p'])
        np.testing.assert_array_equal(result['t'], original['t'])
        np.testing.assert_array_equal(self.events, original)

    def test_stale_rotation_setting_rejected_before_opening_file(self):
        with patch.object(replay.Config, 'get_alignment', return_value=dict(replay_apply_rotation=True)), \
             patch.object(replay, '_build_events_iterator') as build:
            with self.assertRaisesRegex(ValueError, "already rotated"):
                replay.load_processed_event_file('processed.raw', {})
        build.assert_not_called()

    def test_cleanup_on_alignment_failure(self):
        cleanup = Mock()
        with patch.object(replay.Config, 'get_alignment', return_value={}), \
             patch.object(replay, '_build_events_iterator', return_value=(Packets([self.events]), cleanup)), \
             patch.object(replay, 'align_to_rgb', side_effect=RuntimeError("alignment failed")):
            with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                replay.load_processed_event_file('processed.raw', dict(delta_t=2000, n_events=10000))
        cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
