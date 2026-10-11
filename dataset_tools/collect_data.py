#!/usr/bin/env python3
"""
Synchronized capture of FLIR RGB frames (.mp4) and event‑camera data (.raw).
Each recording keeps one index until both files are saved. Quit requests finish
the current recording before workers exit; stalled workers have a bounded grace
period. Event preview rotation does not change the device's RAW log.

Usage:
  python3 dataset_tools/collect_data.py
  Controls: r=start, f=finish/save, q=save and quit (also Ctrl+C).
"""


import sys
import time
import signal
import argparse
from pathlib import Path
from multiprocessing import Process, shared_memory, set_start_method

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import SCENES, stage_path

if __package__:
    from .recording_session import RecordingSession, wait_for_shutdown
else:
    from recording_session import RecordingSession, wait_for_shutdown

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

RGB_SIZE = (1080, 1440)  # (w, h) after 90° rotate

TARGET_FPS = 50
FRAME_PERIOD = 1.0 / TARGET_FPS
SHUTDOWN_TIMEOUT = 60.0  # Seconds allowed for both camera workers to save and exit.

EVENT_SHAPE = (1_000_000,)  # buffer for ~500k events per render cycle
EVENT_DTYPE = {
    "names": ["x", "y", "p", "t"],
    "formats": ["<u2", "<u2", "<i2", "<i8"],
    "offsets": [0, 2, 4, 8],
    "itemsize": 16,
}
FLIR_SHAPE = (RGB_SIZE[1], RGB_SIZE[0], 3)  # (H, W, C) = (1440,1080,3)
FLIR_DTYPE = np.uint8


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def next_record_index(save_dir=None) -> int:
    """Return the next integer index so that event/idx.raw and rgb/idx.mp4 match."""
    save_dir = stage_path("raw", explicit=save_dir)
    existing = [p.stem for p in (save_dir / "event").glob("*.raw")] + [p.stem for p in (save_dir / "rgb").glob("*.mp4")]
    nums = [int(s) for s in existing if s.isdigit()]
    return 0 if not nums else max(nums) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard listener
# ──────────────────────────────────────────────────────────────────────────────

def keyboard_listener(session):
    from pynput import keyboard as kb
    import pyttsx3

    tts = pyttsx3.init()
    tts.say("System ready. Press r to record, f to finish, q to quit.")
    tts.runAndWait()

    def on_press(key):
        try:
            if key.char == "r":
                index = session.start()
                if index is None:
                    print("[RECORD] Still recording or saving; finish and wait before starting again.")
                    return
                tts.say(f"Recording {index}. Throw the ball now."); tts.runAndWait()
            elif key.char == "f":
                if session.finish():
                    tts.say("Finish recording."); tts.runAndWait()
            elif key.char == "q":
                session.request_shutdown()
                return False
        except AttributeError:
            pass  # non‑char key

    with kb.Listener(on_press=on_press) as listener:
        while listener.is_alive() and not session.shutdown_requested.wait(0.1):
            pass


# ──────────────────────────────────────────────────────────────────────────────
# FLIR capture & record
# ──────────────────────────────────────────────────────────────────────────────

def save_rgb_frames(index, frames, save_dir=None):
    """Close the video before the worker acknowledges a saved recording."""
    if not frames:
        raise RuntimeError(f"RGB recording {index} has no frames; pair was not saved completely")
    out_path = stage_path("raw", explicit=save_dir) / "rgb" / f"{index}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, RGB_SIZE)
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {out_path}")
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    print(f"[FLIR] Saved {out_path} – {len(frames)} frames", flush=True)


def record_flir(shared_flir_meta, session, save_dir=None):
    import PySpin

    shm_mem, shm_dtype, shm_shape = shared_flir_meta
    shm    = shared_memory.SharedMemory(name=shm_mem.name)
    shm_np = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)

    system   = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        shm.close()
        raise RuntimeError("No FLIR camera detected")

    cam = cam_list[0]
    cam.Init()

    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    # Enable writable framerate node if available
    if hasattr(cam, "AcquisitionFrameRateEnable") and PySpin.IsWritable(cam.AcquisitionFrameRateEnable):
        cam.AcquisitionFrameRateEnable.SetValue(True)

    if hasattr(cam, "AcquisitionFrameRate") and PySpin.IsWritable(cam.AcquisitionFrameRate):
        cam.AcquisitionFrameRate.SetValue(TARGET_FPS)
        actual = cam.AcquisitionFrameRate.GetValue()
        print(f"[FLIR] Frame‑rate set to {actual:.2f} fps")
    else:
        default_fps = None
        if hasattr(cam, "AcquisitionFrameRate") and PySpin.IsReadable(cam.AcquisitionFrameRate):
            default_fps = cam.AcquisitionFrameRate.GetValue()
        msg = "⚠ Could not set frame‑rate; using camera default"
        if default_fps:
            msg += f" ({default_fps:.2f} fps)"
        print(msg)

    if hasattr(cam.TLStream, "StreamBufferHandlingMode"):
        cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)

    cam.BeginAcquisition()

    frame_buffer = []
    current_index = None
    last_index = None
    next_frame_time  = time.perf_counter()

    def finish_recording():
        nonlocal current_index, last_index, frame_buffer
        index = current_index
        current_index = None  # Do not retry a failed save from the finally block.
        save_rgb_frames(index, frame_buffer, save_dir)
        frame_buffer = []
        last_index = index
        session.saved("rgb", index)

    try:
        while True:
            if current_index is None:
                current_index = session.pending_index(last_index)
                if current_index is not None:
                    frame_buffer = []
                    print(f"[FLIR] Recording started (index {current_index})", flush=True)

            if current_index is not None and session.finish_requested.is_set():
                finish_recording()
            if session.shutdown_requested.is_set():
                break

            image_result = cam.GetNextImage(1000)
            try:
                if image_result.IsIncomplete():
                    continue
                frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_BAYER_RG2RGB)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                shm_np[:] = frame
                if current_index is not None:
                    frame_buffer.append(frame.copy())
            finally:
                image_result.Release()

            # Maintain pacing
            next_frame_time += FRAME_PERIOD
            sleep = next_frame_time - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_frame_time = time.perf_counter()

    finally:
        try:
            if current_index is not None:
                finish_recording()
        finally:
            cam.EndAcquisition(); cam.DeInit(); cam_list.Clear(); system.ReleaseInstance(); shm.close()


# ──────────────────────────────────────────────────────────────────────────────
# Event‑camera capture & record
# ──────────────────────────────────────────────────────────────────────────────

def record_event(shared_event_meta, session, save_dir=None):
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_core.event_io import EventsIterator

    ev_mem, ev_dtype_desc, ev_shape = shared_event_meta
    shm = shared_memory.SharedMemory(name=ev_mem.name)
    shm_np = np.ndarray(ev_shape, dtype=np.dtype(ev_dtype_desc), buffer=shm.buf)

    dev = initiate_device(""); iterator = iter(EventsIterator.from_device(device=dev))
    current_index = None
    last_index = None

    def finish_recording():
        nonlocal current_index, last_index
        index = current_index
        current_index = None
        dev.get_i_events_stream().stop_log_raw_data()
        last_index = index
        session.saved("event", index)
        print(f"[EVENT] Saved recording {index}", flush=True)

    try:
        while True:
            if current_index is None:
                index = session.pending_index(last_index)
                if index is not None:
                    out_path = stage_path("raw", explicit=save_dir) / "event" / f"{index}.raw"
                    dev.get_i_events_stream().log_raw_data(str(out_path))
                    current_index = index
                    print(f"[EVENT] Recording started: {out_path}", flush=True)

            if current_index is not None and session.finish_requested.is_set():
                finish_recording()
            if session.shutdown_requested.is_set():
                break

            ev = next(iterator)
            ev["x"], ev["y"] = ev["y"].copy(), -ev["x"].copy(); ev["y"] += 1279

            event=ev[:1000000]
            # copy ONLY the portion we have; clears rest via length slice – cheaper than full memset
            n = event.shape[0]
            shm_np[:n] = event
            shm_np[n:] = 0  # keep buffer clean for viewer

    finally:
        try:
            if current_index is not None:
                finish_recording()
        finally:
            shm.close()


# ──────────────────────────────────────────────────────────────────────────────
# Simple visualizer (optional)
# ──────────────────────────────────────────────────────────────────────────────

def show_streams(shared_event_meta, shared_flir_meta, session):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    ev_mem, ev_dtype_desc, ev_shape = shared_event_meta
    flir_mem, flir_dtype, flir_shape = shared_flir_meta

    ev_shm = shared_memory.SharedMemory(name=ev_mem.name)
    flir_shm = shared_memory.SharedMemory(name=flir_mem.name)

    ev_np = np.ndarray(ev_shape, dtype=np.dtype(ev_dtype_desc), buffer=ev_shm.buf)
    flir_np = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    gen = PeriodicFrameGenerationAlgorithm(sensor_width=720, sensor_height=1280, fps=100, palette=ColorPalette.Dark)
    cd_frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    while not session.shutdown_requested.is_set():
        gen.generate_frame(ev_np, cd_frame, accumulation_time_us=1000)

        # Resize both frames to the same size for overlay
        target_size = (int(RGB_SIZE[0] * 0.6), int(RGB_SIZE[1] * 0.6))
        rgb_resized = cv2.resize(flir_np, target_size)

        # Resize event frame to match RGB (720x1280 -> 1080x1440 then resize)
        # Event frame needs to be resized to match RGB dimensions
        evt_resized_to_rgb = cv2.resize(cd_frame, (RGB_SIZE[0], RGB_SIZE[1]))
        evt_resized = cv2.resize(evt_resized_to_rgb, target_size)

        # Create mask: detect non-black pixels in event frame
        # Convert to grayscale and threshold to find where events exist
        evt_gray = cv2.cvtColor(evt_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(evt_gray, 1, 255, cv2.THRESH_BINARY)

        # Invert mask for areas without events
        mask_inv = cv2.bitwise_not(mask)

        # Convert mask to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR) / 255.0

        # Blend: RGB in non-event areas, RGB + dimmed events in event areas
        evt_dimmed = (evt_resized * 0.5).astype(np.uint8)  # Reduce event brightness to 50%
        overlay = (rgb_resized * mask_inv_3ch +
                   (rgb_resized * 0.7 + evt_dimmed * 0.3) * mask_3ch).astype(np.uint8)

        cv2.imshow("RGB + Event Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.request_shutdown()
            break

    ev_shm.close(); flir_shm.close(); cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run_worker(target, args):
    # The parent handles terminal Ctrl+C and gives camera writers time to close.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    target(*args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=SCENES, default='normal')
    parser.add_argument('--save-dir', '--output-dir', dest='save_dir',
                        help='Raw recording output (default: machine data_root/saved_data_raw/saved_data_<scene>)')
    args = parser.parse_args(argv)
    try:
        args.save_dir = stage_path('raw', args.task, args.save_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    save_dir = args.save_dir
    print(f"Capture output: {save_dir}", flush=True)
    (save_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (save_dir / 'event').mkdir(parents=True, exist_ok=True)

    # Shared memory
    event_mem = shared_memory.SharedMemory(create=True, size=np.dtype(EVENT_DTYPE).itemsize * EVENT_SHAPE[0])
    flir_mem = shared_memory.SharedMemory(create=True, size=np.prod(FLIR_SHAPE) * np.dtype(FLIR_DTYPE).itemsize)
    shared_event_meta = (event_mem, EVENT_DTYPE, EVENT_SHAPE)
    shared_flir_meta = (flir_mem, FLIR_DTYPE, FLIR_SHAPE)

    # Sync primitives
    session = RecordingSession(next_record_index(save_dir))

    def request_exit(sig, frame):
        print("\n[MAIN] Finishing current recording before exit...", flush=True)
        session.request_shutdown()

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)

    workers = [
        ("rgb", record_flir, (shared_flir_meta, session, save_dir)),
        ("event", record_event, (shared_event_meta, session, save_dir)),
        ("keyboard", keyboard_listener, (session,)),
        ("viewer", show_streams, (shared_event_meta, shared_flir_meta, session)),
    ]
    procs = []
    failed = False
    try:
        for name, target, args in workers:
            process = Process(name=name, target=run_worker, args=(target, args))
            process.start()
            procs.append(process)
        while not session.shutdown_requested.wait(0.2):
            exited = [p for p in procs if p.exitcode is not None]
            if exited:
                print(f"[MAIN] Worker exited unexpectedly: {[(p.name, p.exitcode) for p in exited]}", flush=True)
                failed = True
                break
    finally:
        session.request_shutdown()
        forced = wait_for_shutdown(procs, SHUTDOWN_TIMEOUT)
        failed = failed or bool(forced) or any(p.exitcode != 0 for p in procs)
        unfinished = session.unfinished_index()
        if unfinished is not None:
            print(f"[MAIN] Recording {unfinished} was not confirmed saved by both cameras.", flush=True)
            failed = True
        event_mem.close(); event_mem.unlink(); flir_mem.close(); flir_mem.unlink()

    print("[MAIN] Shutdown incomplete; check recordings." if failed else "[MAIN] Shutdown complete.")
    return 1 if failed else 0


if __name__ == "__main__":
    set_start_method("spawn")
    sys.exit(main())
