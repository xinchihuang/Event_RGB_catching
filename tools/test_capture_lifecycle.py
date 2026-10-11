"""Recording identity and shutdown tests without cameras, GUI or camera SDKs."""

import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_tools import collect_data as capture
from dataset_tools.recording_session import RecordingSession, wait_for_shutdown


def delayed_save(session, ready, release, result):
    """Simulate the RGB writer completing later than the RAW logger."""
    index = session.pending_index(None)
    ready.set()
    release.wait(5)
    session.saved("rgb", index)
    result.put(index)


class SessionTests(unittest.TestCase):
    def test_index_waits_for_both_saves_even_with_delayed_process(self):
        context = multiprocessing.get_context("spawn")
        session = RecordingSession(5, context)
        self.assertEqual(session.start(), 5)
        ready, release, result = context.Event(), context.Event(), context.Queue()
        process = context.Process(target=delayed_save, args=(session, ready, release, result))
        process.start()
        try:
            self.assertTrue(ready.wait(5))
            session.finish()
            session.saved("event", 5)
            session.saved("event", 5)  # Duplicate confirmation cannot advance twice.
            self.assertIsNone(session.start())
            self.assertEqual(session.unfinished_index(), 5)
            release.set()
            self.assertEqual(result.get(timeout=5), 5)
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            self.assertIsNone(session.unfinished_index())
            self.assertEqual(session.start(), 6)
            session.saved("rgb", 5)  # A stale acknowledgement cannot close record 6.
            self.assertEqual(session.unfinished_index(), 6)
        finally:
            release.set()
            if process.is_alive():
                process.terminate()
                process.join(2)
            result.close()
            result.join_thread()

    def test_fast_finish_and_quit_preserve_unseen_record(self):
        session = RecordingSession(0)
        self.assertFalse(session.finish())
        self.assertEqual(session.start(), 0)
        session.finish()
        session.request_shutdown()
        self.assertEqual(session.pending_index(None), 0)
        self.assertTrue(session.finish_requested.is_set())
        self.assertIsNone(session.start())
        session.saved("rgb", 0)
        self.assertEqual(session.pending_index(None), 0)
        session.saved("event", 0)
        self.assertIsNone(session.unfinished_index())
        self.assertIsNone(session.start())

    def test_idle_quit_does_not_create_a_record(self):
        session = RecordingSession(5)
        session.request_shutdown()
        self.assertIsNone(session.pending_index(None))
        self.assertIsNone(session.unfinished_index())


class CaptureWorkerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'EVENT_RGB_DATA_ROOT': '/tmp/capture-test-data'})
        environment.start()
        self.addCleanup(environment.stop)

    def flir_mocks(self):
        camera = Mock()
        cameras = Mock()
        cameras.GetSize.return_value = 1
        cameras.__getitem__ = Mock(return_value=camera)
        system = Mock()
        system.GetCameras.return_value = cameras
        sdk = types.SimpleNamespace(
            System=types.SimpleNamespace(GetInstance=lambda: system),
            IsWritable=lambda node: False, IsReadable=lambda node: False,
            AcquisitionMode_Continuous=1, StreamBufferHandlingMode_NewestOnly=1,
        )
        memory = Mock(buf=bytearray(12))
        image = Mock()
        image.IsIncomplete.return_value = False
        image.GetNDArray.return_value = np.zeros((2, 2), dtype=np.uint8)
        return sdk, camera, system, memory, image

    def run_flir(self, session, on_image, writer):
        sdk, camera, system, memory, image = self.flir_mocks()
        camera.GetNextImage.side_effect = lambda timeout: on_image(image)
        meta = (types.SimpleNamespace(name="mock-rgb"), np.uint8, (2, 2, 3))
        with patch.dict(sys.modules, {"PySpin": sdk}), \
             patch.object(capture.shared_memory, "SharedMemory", return_value=memory), \
             patch.object(capture.cv2, "cvtColor", return_value=np.zeros((2, 2, 3), dtype=np.uint8)), \
             patch.object(capture.cv2, "rotate", side_effect=lambda frame, _: frame), \
             patch.object(capture.cv2, "VideoWriter", return_value=writer) as factory:
            capture.record_flir(meta, session)
        return camera, memory, image, factory

    def test_rgb_quit_flushes_frames_to_reserved_index_before_ack(self):
        session = RecordingSession(5)
        session.start()
        writer = Mock()
        writer.isOpened.return_value = True

        def on_image(image):
            session.request_shutdown()  # Quit arrives while the image is in flight.
            session.saved("event", 5)
            return image

        writer.release.side_effect = lambda: self.assertEqual(session.unfinished_index(), 5)
        camera, memory, image, factory = self.run_flir(session, on_image, writer)
        self.assertTrue(factory.call_args.args[0].endswith("/rgb/5.mp4"))
        writer.write.assert_called_once()
        writer.release.assert_called_once()
        image.Release.assert_called_once()
        camera.EndAcquisition.assert_called_once()
        memory.close.assert_called_once()
        self.assertIsNone(session.unfinished_index())

    def test_rgb_exception_still_flushes_buffered_frames(self):
        session = RecordingSession(9)
        session.start()
        writer = Mock()
        writer.isOpened.return_value = True
        calls = 0

        def on_image(image):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("camera disconnected")
            return image

        with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
            self.run_flir(session, on_image, writer)
        writer.write.assert_called_once()
        writer.release.assert_called_once()
        # The event worker has not confirmed saving, so the index stays reserved.
        self.assertEqual(session.unfinished_index(), 9)

    def test_rgb_writer_failure_does_not_acknowledge_or_retry(self):
        session = RecordingSession(3)
        session.start()
        writer = Mock()
        writer.isOpened.return_value = False

        def on_image(image):
            session.request_shutdown()
            session.saved("event", 3)
            return image

        with self.assertRaisesRegex(RuntimeError, "Cannot open video writer"):
            self.run_flir(session, on_image, writer)
        writer.release.assert_called_once()
        self.assertEqual(session.unfinished_index(), 3)

    def test_quit_before_first_rgb_frame_reports_incomplete_pair(self):
        session = RecordingSession(3)
        session.start()
        session.request_shutdown()
        writer = Mock()
        with self.assertRaisesRegex(RuntimeError, "no frames"):
            self.run_flir(session, lambda image: image, writer)
        writer.write.assert_not_called()
        self.assertEqual(session.unfinished_index(), 3)

    def test_raw_logging_stops_before_save_ack_on_quit(self):
        session = RecordingSession(5)
        session.start()
        stream = Mock()
        device = Mock()
        device.get_i_events_stream.return_value = stream
        events = np.array([(1, 2, 1, 10)], dtype=np.dtype(capture.EVENT_DTYPE))

        def packets():
            session.request_shutdown()
            session.saved("rgb", 5)
            yield events

        event_io = types.ModuleType("metavision_core.event_io")
        event_io.EventsIterator = types.SimpleNamespace(from_device=lambda device: packets())
        raw_reader = types.ModuleType("metavision_core.event_io.raw_reader")
        raw_reader.initiate_device = lambda path: device
        modules = {"metavision_core": types.ModuleType("metavision_core"),
                   "metavision_core.event_io": event_io,
                   "metavision_core.event_io.raw_reader": raw_reader}
        memory = Mock(buf=bytearray(events.nbytes * 4))
        meta = (types.SimpleNamespace(name="mock-event"), capture.EVENT_DTYPE, (4,))
        stream.stop_log_raw_data.side_effect = lambda: self.assertEqual(session.unfinished_index(), 5)
        with patch.dict(sys.modules, modules), \
             patch.object(capture.shared_memory, "SharedMemory", return_value=memory):
            capture.record_event(meta, session)
        self.assertTrue(stream.log_raw_data.call_args.args[0].endswith("/event/5.raw"))
        stream.stop_log_raw_data.assert_called_once()
        memory.close.assert_called_once()
        self.assertIsNone(session.unfinished_index())


class ShutdownTests(unittest.TestCase):
    def test_grace_period_waits_for_slow_save_without_terminate(self):
        context = multiprocessing.get_context("spawn")
        session = RecordingSession(5, context)
        session.start()
        ready, release, result = context.Event(), context.Event(), context.Queue()
        process = context.Process(target=delayed_save, args=(session, ready, release, result))
        process.start()
        timer = None
        try:
            self.assertTrue(ready.wait(5))
            session.request_shutdown()
            session.saved("event", 5)
            timer = threading.Timer(0.1, release.set)
            timer.start()
            self.assertEqual(wait_for_shutdown([process], 5), [])
            self.assertEqual(result.get(timeout=1), 5)
            self.assertEqual(process.exitcode, 0)
            self.assertIsNone(session.unfinished_index())
        finally:
            if timer:
                timer.join()
            if process.is_alive():
                process.terminate()
                process.join(2)
            result.close()
            result.join_thread()

    def test_stuck_worker_is_reported_and_terminated_after_deadline(self):
        context = multiprocessing.get_context("spawn")
        process = context.Process(name="stuck-camera", target=time.sleep, args=(30,))
        process.start()
        try:
            self.assertEqual(wait_for_shutdown([process], 0.1), ["stuck-camera"])
            self.assertFalse(process.is_alive())
            self.assertNotEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join(2)


if __name__ == "__main__":
    unittest.main()
