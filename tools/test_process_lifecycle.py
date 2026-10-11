"""Fault injection without camera SDKs, GPU, motors or GUI windows."""
import ast
import multiprocessing
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from process_lifecycle import run_workers, shutdown_processes


PROJECT = Path(__file__).resolve().parents[1]


def crash():
    sys.exit(7)


def wait_for_stop(stop, stopped):
    stop.wait(10)
    if stop.is_set():
        stopped.set()


def crash_during_shutdown(stop):
    stop.wait(10)
    sys.exit(9)


def ignore_stop(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    time.sleep(30)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.ctx = multiprocessing.get_context('spawn')
        self.stop = self.ctx.Event()
        self.state = self.ctx.Value('u', 't')
        self.workers = []

    def tearDown(self):
        for process in self.workers:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(timeout=3)

    def worker(self, target, *args):
        process = self.ctx.Process(target=target, args=args)
        self.workers.append(process)
        return process

    def test_crash_stops_peer_and_returns_failure(self):
        stopped = self.ctx.Event()
        peer = self.worker(wait_for_stop, self.stop, stopped)
        code = run_workers([('camera', self.worker(crash)), ('peer', peer)],
                           self.stop, self.state)
        self.assertEqual(code, 1)
        self.assertTrue(stopped.is_set())
        self.assertEqual(peer.exitcode, 0)
        self.assertEqual(self.state.value, 'e')
        self.assertTrue(all(not p.is_alive() for p in self.workers))

    def test_eof_and_user_stop_are_successful(self):
        for reason in ('eof', 'state', 'event'):
            with self.subTest(reason=reason):
                self.stop.clear()
                self.state.value = 't'
                stopped = self.ctx.Event()
                peer = self.worker(wait_for_stop, self.stop, stopped)

                def finished():
                    if reason == 'state':
                        self.state.value = 'e'
                    if reason == 'event':
                        self.stop.set()
                    return reason == 'eof'

                self.assertEqual(run_workers([('peer', peer)], self.stop, self.state,
                                             finished=finished), 0)
                self.assertTrue(stopped.is_set())
                self.assertEqual(peer.exitcode, 0)

    def test_crash_concurrent_with_eof_is_not_success(self):
        worker = self.worker(crash_during_shutdown, self.stop)
        self.assertEqual(run_workers([('late_crash', worker)], self.stop, self.state,
                                     finished=lambda: True), 1)
        self.assertEqual(worker.exitcode, 9)

    def test_startup_error_still_stops_started_workers(self):
        stopped = self.ctx.Event()
        peer = self.worker(wait_for_stop, self.stop, stopped)
        broken = Mock()
        broken.start.side_effect = RuntimeError('cannot start')
        with self.assertRaisesRegex(RuntimeError, 'cannot start'):
            run_workers([('peer', peer), ('broken', broken)], self.stop, self.state)
        self.assertTrue(stopped.is_set())
        self.assertEqual(peer.exitcode, 0)

    def test_keyboard_interrupt_stops_workers(self):
        stopped = self.ctx.Event()

        def interrupted():
            raise KeyboardInterrupt

        self.assertEqual(run_workers(
            [('peer', self.worker(wait_for_stop, self.stop, stopped))],
            self.stop, self.state, finished=interrupted), 0)
        self.assertTrue(stopped.is_set())

    def test_signal_handlers_only_request_stop(self):
        for filename in ('tracking.py', 'live_real.py', 'record_compare.py'):
            with self.subTest(filename=filename):
                self.stop.clear()
                tree = ast.parse((PROJECT / filename).read_text())
                handler = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                               and n.name == 'signal_handler')
                namespace = dict(stop_event=self.stop)
                exec(compile(ast.Module(body=[handler], type_ignores=[]), filename, 'exec'),
                     namespace)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    namespace['signal_handler'](sig, None)
                    self.assertTrue(self.stop.is_set())

    def test_compare_starter_does_not_restart_after_stop(self):
        tree = ast.parse((PROJECT / 'record_compare.py').read_text())
        starter = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'starter')
        namespace = dict(time=time)
        exec(compile(ast.Module(body=[starter], type_ignores=[]), 'record_compare.py', 'exec'),
             namespace)
        self.stop.set()
        self.state.value = 'e'
        namespace['starter'](self.state, self.stop)
        self.assertEqual(self.state.value, 'e')

    def test_timeout_escalates_to_kill_and_reaps(self):
        ready = self.ctx.Event()
        stuck = self.worker(ignore_stop, ready)
        stuck.start()
        self.assertTrue(ready.wait(5))
        forced = shutdown_processes([stuck], timeout=0.05, terminate_timeout=0.1)
        self.assertEqual(forced, [stuck.name])
        self.assertEqual(stuck.exitcode, -signal.SIGKILL)
        self.assertFalse(stuck.is_alive())

    def test_forced_shutdown_returns_failure(self):
        ready = self.ctx.Event()
        stuck = self.worker(ignore_stop, ready)
        self.assertEqual(run_workers([('stuck', stuck)], self.stop, self.state,
                                     finished=lambda: ready.wait(5),
                                     shutdown_timeout=0.05), 1)
        self.assertFalse(stuck.is_alive())

    def entrypoint_tail(self, filename, process, save=None):
        """Execute the actual main shutdown block with hardware setup replaced."""
        source = (PROJECT / filename).read_text()
        module = ast.parse(source)
        main = next(n for n in module.body if isinstance(n, ast.If)
                    and any(isinstance(part, ast.Name) and part.id == '__name__'
                            for part in ast.walk(n.test)))
        index = next(i for i, n in enumerate(main.body)
                     if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'exit_code'
                             for t in n.targets))
        block = ast.Module(body=main.body[index:], type_ignores=[])
        memory_names = ('event_memory', 'flir_memory', 'ROI_memory', 'empty_ROI_memory',
                        'ball_pixel_memory', 'ball_pixel_3D_memory', 'ball_pixel_3D_raw_memory',
                        'ball_pixel_3D_rgb_memory', 'impact_point_memory')
        memory = {name: Mock() for name in memory_names}
        manager = Mock()
        save = save or Mock()
        namespace = dict(memory, sys=sys, run_workers=run_workers, USE_GUI=False,
                         processes_to_start=['test_worker'], all_processes={'test_worker': process},
                         stop_event=self.stop, system_state=self.state, shm_manager=manager,
                         event_broadcasting=SimpleNamespace(value='f'),
                         video_broadcasting=SimpleNamespace(value='f'),
                         args=SimpleNamespace(test_mode='Event', task='normal'),
                         ball_pixel_3D=SimpleNamespace(shape=(10, 4)),
                         ball_observations=SimpleNamespace(value=1),
                         record_index=SimpleNamespace(value=0), output_root='unused',
                         save_tracking_results=save)
        try:
            exec(compile(block, filename, 'exec'), namespace)
        finally:
            if filename == 'record_compare.py':
                for shm in memory.values():
                    shm.close.assert_called_once()
                    shm.unlink.assert_called_once()
            else:
                manager.cleanup_all.assert_called_once()
        return save

    def test_all_three_entrypoints_propagate_worker_failure_and_cleanup(self):
        for filename in ('tracking.py', 'live_real.py', 'record_compare.py'):
            with self.subTest(filename=filename):
                self.stop.clear()
                self.state.value = 't'
                save = Mock()
                with self.assertRaises(SystemExit) as exc:
                    self.entrypoint_tail(filename, self.worker(crash), save)
                self.assertEqual(exc.exception.code, 1)
                save.assert_not_called()

    def test_all_three_entrypoints_cleanup_on_startup_error(self):
        for filename in ('tracking.py', 'live_real.py', 'record_compare.py'):
            with self.subTest(filename=filename):
                self.stop.clear()
                self.state.value = 't'
                broken = Mock()
                broken.start.side_effect = RuntimeError('startup failed')
                with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                    self.entrypoint_tail(filename, broken)

    def test_compare_save_error_still_releases_shared_memory(self):
        stopped = self.ctx.Event()
        save = Mock(side_effect=OSError('disk full'))
        with self.assertRaisesRegex(OSError, 'disk full'):
            self.entrypoint_tail('record_compare.py',
                                 self.worker(wait_for_stop, self.stop, stopped), save)
        save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
