"""Parent-side worker supervision for live tracking and comparison replay.

Workers share a stop event; legacy keyboard workers also need state='e'.
Callers own shared memory and must release it after run_workers returns/raises.
"""
import time


def failed_workers(processes):
    return [p for p in processes if p.exitcode not in (None, 0)]


def shutdown_processes(processes, timeout=5, terminate_timeout=1):
    """Join workers under one deadline, then terminate/kill stragglers.

    Returns names requiring forced shutdown. The caller must signal stop first.
    """
    def join_until_deadline(workers, seconds):
        deadline = time.monotonic() + seconds
        for process in workers:
            process.join(timeout=max(0, deadline - time.monotonic()))

    join_until_deadline(processes, timeout)
    remaining = [p for p in processes if p.is_alive()]
    forced = [p.name for p in remaining]
    for process in remaining:
        print(f"Shutdown timeout: terminating {process.name}", flush=True)
        process.terminate()
    join_until_deadline(remaining, terminate_timeout)
    stubborn = [p for p in remaining if p.is_alive()]
    for process in stubborn:
        print(f"Still running: killing {process.name}", flush=True)
        process.kill()
    join_until_deadline(stubborn, terminate_timeout)
    for process in stubborn:
        if process.is_alive():
            raise RuntimeError(f"Worker could not be stopped: {process.name}")
    return forced


def run_workers(named_processes, stop_event, system_state, finished=None,
                shutdown_timeout=5):
    """Start and supervise workers; return 1 for a crash or forced shutdown.

    finished optionally detects replay EOF. Check exit codes again after joining
    so a crash concurrent with EOF or a user stop cannot be reported as success.
    Startup exceptions propagate after stopping all workers already started.
    """
    started = []
    try:
        for name, process in named_processes:
            if stop_event.is_set() or system_state.value == 'e':
                break
            process.name = name
            process.start()
            started.append(process)
            print(f"Started: {name}", flush=True)

        while True:
            if failed_workers(started):
                break
            if stop_event.is_set() or system_state.value == 'e':
                break
            if finished is not None and finished():
                break
            if not any(p.is_alive() for p in started):
                break
            stop_event.wait(0.1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...", flush=True)
    finally:
        stop_event.set()
        system_state.value = 'e'
        forced = shutdown_processes(started, timeout=shutdown_timeout)

    failed = failed_workers(started)
    if failed:
        details = ", ".join(f"{p.name} (exit code {p.exitcode})" for p in failed)
        print(f"Subprocess failure: {details}", flush=True)
    return 1 if failed or forced else 0
