#!/usr/bin/env python3
"""
Synchronized capture of FLIR RGB frames (.mp4) and event‑camera data (.raw).
This revision fixes two regressions in the version you pasted back:
1. **Event shared‑memory copy was commented out** – restored so the viewer actually
   sees live events.
2. **Viewer double‑processing** – now uses only `generate_frame()` for CD frames.
Other minor clean‑ups are annotated inline.
"""


import os
import sys
import time
import signal
from pathlib import Path
from multiprocessing import Process, Event, Value, shared_memory, set_start_method

import numpy as np
import cv2
import pyttsx3
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("/home/xinchi/Event_RGB_catching_data/saved_data")
EVENT_DIR = SAVE_DIR / "event"
RGB_DIR = SAVE_DIR / "rgb"
RGB_DIR.mkdir(parents=True, exist_ok=True)
EVENT_DIR.mkdir(parents=True, exist_ok=True)

RGB_SIZE = (1080, 1440)  # (w, h) after 90° rotate

TARGET_FPS = 50                              # ← 100 fps everywhere
FRAME_PERIOD = 1.0 / TARGET_FPS

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

def next_record_index() -> int:
    """Return the next integer index so that event/idx.raw and rgb/idx.mp4 match."""
    existing = [p.stem for p in EVENT_DIR.glob("*.raw")] + [p.stem for p in RGB_DIR.glob("*.mp4")]
    nums = [int(s) for s in existing if s.isdigit()]
    return 0 if not nums else max(nums) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard listener
# ──────────────────────────────────────────────────────────────────────────────

def keyboard_listener(evt_start: Event, evt_stop: Event, stop_evt: Event, idx_val: Value):
    from pynput import keyboard as kb

    tts = pyttsx3.init()
    tts.say("System ready. Press r to record, f to finish, q to quit.")
    tts.runAndWait()

    def on_press(key):
        try:
            if key.char == "r":
                evt_start.set(); evt_stop.clear()
                tts.say(f"Recording {idx_val.value}. Throw the ball now."); tts.runAndWait()
            elif key.char == "f":
                evt_stop.set(); evt_start.clear()
                tts.say("Finish recording."); tts.runAndWait()
                with idx_val.get_lock():
                    idx_val.value += 1  # next index
            elif key.char == "q":
                stop_evt.set(); return False  # exit listener
        except AttributeError:
            pass  # non‑char key

    with kb.Listener(on_press=on_press) as listener:
        listener.join()


# ──────────────────────────────────────────────────────────────────────────────
# FLIR capture & record
# ──────────────────────────────────────────────────────────────────────────────

def record_flir(shared_flir_meta, evt_start: Event, evt_stop: Event, idx_val: Value, stop_evt: Event):
    import PySpin

    shm_mem, shm_dtype, shm_shape = shared_flir_meta
    shm    = shared_memory.SharedMemory(name=shm_mem.name)
    shm_np = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)

    system   = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("[FLIR] No camera detected."); return

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    frames_written   = 0
    clip_start_time  = None
    next_frame_time  = time.perf_counter()

    try:
        while not stop_evt.is_set():
            image_result = cam.GetNextImage(1000)
            if image_result.IsIncomplete():
                continue

            frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_BAYER_RG2RGB)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            shm_np[:] = frame

            # Handle recording start
            if evt_start.is_set() and writer is None:
                out_path = RGB_DIR / f"{idx_val.value}.mp4"
                writer   = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS,RGB_SIZE)
                frames_written  = 0
                clip_start_time = time.perf_counter()
                print(f"[FLIR] ▶ {out_path}")

            # Write frame when recording
            if writer and not evt_stop.is_set():
                writer.write(frame)
                frames_written += 1

            # Stop recording
            if evt_stop.is_set() and writer:
                writer.release(); writer = None
                duration = time.perf_counter() - clip_start_time if clip_start_time else frames_written / TARGET_FPS
                avg_fps  = frames_written / duration if duration > 0 else 0
                print(f"[FLIR] ■ Clip saved – {frames_written} frames, {duration:.2f}s (≈{avg_fps:.2f} fps)")

            # Maintain pacing
            next_frame_time += FRAME_PERIOD
            sleep = next_frame_time - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_frame_time = time.perf_counter()

            image_result.Release()
    finally:
        if writer:
            writer.release()
        cam.EndAcquisition(); cam.DeInit(); cam_list.Clear(); system.ReleaseInstance(); shm.close()


# ──────────────────────────────────────────────────────────────────────────────
# Event‑camera capture & record
# ──────────────────────────────────────────────────────────────────────────────

def record_event(shared_event_meta, evt_start: Event, evt_stop: Event, idx_val: Value, stop_evt: Event):
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_core.event_io import EventsIterator

    ev_mem, ev_dtype_desc, ev_shape = shared_event_meta
    shm = shared_memory.SharedMemory(name=ev_mem.name)
    shm_np = np.ndarray(ev_shape, dtype=np.dtype(ev_dtype_desc), buffer=shm.buf)

    dev = initiate_device(""); iterator = iter(EventsIterator.from_device(device=dev))
    recording = False; start_ts = None

    try:
        while not stop_evt.is_set():
            ev = next(iterator)
            ev["x"], ev["y"] = ev["y"].copy(), -ev["x"].copy(); ev["y"] += 1279

            event=ev[:1000000]
            # copy ONLY the portion we have; clears rest via length slice – cheaper than full memset
            n = event.shape[0]
            shm_np[:n] = event
            shm_np[n:] = 0  # keep buffer clean for viewer

            if evt_start.is_set() and not recording:
                out_path = EVENT_DIR / f"{idx_val.value}.raw"
                dev.get_i_events_stream().log_raw_data(str(out_path))
                recording = True; start_ts = time.time(); print(f"[EVENT] ▶ {out_path}")
            if evt_stop.is_set() and recording:
                dev.get_i_events_stream().stop_log_raw_data(); recording = False
                print(f"[EVENT] ■ {time.time() - start_ts:.2f}s")
    finally:
        if recording: dev.get_i_events_stream().stop_log_raw_data()
        shm.close()


# ──────────────────────────────────────────────────────────────────────────────
# Simple visualizer (optional)
# ──────────────────────────────────────────────────────────────────────────────

def show_streams(shared_event_meta, shared_flir_meta, stop_evt: Event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    ev_mem, ev_dtype_desc, ev_shape = shared_event_meta
    flir_mem, flir_dtype, flir_shape = shared_flir_meta

    ev_shm = shared_memory.SharedMemory(name=ev_mem.name)
    flir_shm = shared_memory.SharedMemory(name=flir_mem.name)

    ev_np = np.ndarray(ev_shape, dtype=np.dtype(ev_dtype_desc), buffer=ev_shm.buf)
    flir_np = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    gen = PeriodicFrameGenerationAlgorithm(sensor_width=720, sensor_height=1280, fps=100, palette=ColorPalette.Dark)
    cd_frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    while not stop_evt.is_set():
        gen.generate_frame(ev_np, cd_frame, accumulation_time_us=1000)
        rgb_small = cv2.resize(flir_np, (int(RGB_SIZE[0] * 0.6), int(RGB_SIZE[1] * 0.6)))
        evt_small = cv2.resize(cd_frame, (int(RGB_SIZE[0] * 0.6), int(RGB_SIZE[1] * 0.6)))
        cv2.imshow("FLIR", rgb_small); cv2.imshow("Event", evt_small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_evt.set(); break

    ev_shm.close(); flir_shm.close(); cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_start_method("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # unused but kept

    # Shared memory
    event_mem = shared_memory.SharedMemory(create=True, size=np.dtype(EVENT_DTYPE).itemsize * EVENT_SHAPE[0])
    flir_mem = shared_memory.SharedMemory(create=True, size=np.prod(FLIR_SHAPE) * np.dtype(FLIR_DTYPE).itemsize)
    shared_event_meta = (event_mem, EVENT_DTYPE, EVENT_SHAPE)
    shared_flir_meta = (flir_mem, FLIR_DTYPE, FLIR_SHAPE)

    # Sync primitives
    stop_evt = Event(); start_evt = Event(); stop_rec_evt = Event()
    idx_val = Value("i", next_record_index())

    signal.signal(signal.SIGINT, lambda sig, fr: (print("\n[MAIN] Ctrl‑C"), stop_evt.set()))

    procs = [
        Process(target=keyboard_listener, args=(start_evt, stop_rec_evt, stop_evt, idx_val)),
        Process(target=record_flir, args=(shared_flir_meta, start_evt, stop_rec_evt, idx_val, stop_evt)),
        Process(target=record_event, args=(shared_event_meta, start_evt, stop_rec_evt, idx_val, stop_evt)),
        Process(target=show_streams, args=(shared_event_meta, shared_flir_meta, stop_evt)),
    ]
    for p in procs: p.start()
    while not stop_evt.is_set(): time.sleep(0.2)
    for p in procs: p.terminate(); p.join()

    event_mem.close(); event_mem.unlink(); flir_mem.close(); flir_mem.unlink()
    print("[MAIN] Shutdown complete.")