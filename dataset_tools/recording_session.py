"""Shared recording commands and save acknowledgements, independent of camera SDKs."""

import multiprocessing
import time


class RecordingSession:
    """Keep an index reserved until both RGB and event files have been closed."""

    def __init__(self, index, context=None):
        context = context or multiprocessing.get_context()
        self._lock = context.RLock()
        self._index = context.Value("i", index, lock=False)
        self._active = context.Value("b", False, lock=False)
        self._saved = context.Array("b", 2, lock=False)
        self.finish_requested = context.Event()
        self.shutdown_requested = context.Event()

    def start(self):
        """Return the reserved index, or None while recording/saving/exiting."""
        with self._lock:
            if self._active.value or self.shutdown_requested.is_set():
                return None
            self._saved[:] = [False, False]
            self.finish_requested.clear()
            self._active.value = True
            return self._index.value

    def finish(self):
        with self._lock:
            if not self._active.value:
                return False
            # Leave the command active so even a delayed worker sees this record.
            self.finish_requested.set()
            return True

    def request_shutdown(self):
        with self._lock:
            self.finish_requested.set()
            self.shutdown_requested.set()

    def pending_index(self, last_index):
        with self._lock:
            if self._active.value and self._index.value != last_index:
                return self._index.value
            return None

    def saved(self, worker, index):
        """Acknowledge only after the writer/RAW stream closes successfully."""
        slot = {"rgb": 0, "event": 1}[worker]
        next_index = None
        with self._lock:
            if not self._active.value or index != self._index.value:
                return
            self._saved[slot] = True
            if all(self._saved):
                self._active.value = False
                self._index.value += 1
                self.finish_requested.clear()
                next_index = self._index.value
        if next_index is not None:
            print(f"[RECORD] Saved pair {index}; next index: {next_index}", flush=True)

    def unfinished_index(self):
        with self._lock:
            return self._index.value if self._active.value else None


def wait_for_shutdown(processes, timeout):
    """Allow a shared grace period, then stop stuck workers; return their names."""
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    stuck = [process for process in processes if process.is_alive()]
    for process in stuck:
        print(f"[MAIN] Shutdown timeout: {process.name}; saving may be incomplete.", flush=True)
        process.terminate()
    for process in stuck:
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
    return [process.name for process in stuck]
