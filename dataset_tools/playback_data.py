#!/usr/bin/env python3
"""
Playback synchronized FLIR RGB video (.mp4) and event-camera data (.raw).
Uses read_from_video.py and read_from_raw.py for data loading,
and show_streams from collect_data.py for visualization.
"""

import os
import sys
import time
import signal
from pathlib import Path
from multiprocessing import Process, Event, Value, shared_memory, set_start_method

import numpy as np
import cv2

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import data reading functions
from data_input.read_from_raw import read_from_raw
from data_input.read_from_video import read_from_video

# ──────────────────────────────────────────────────────────────────────────────
# Config (same as collect_data.py)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/xinchi/Event_RGB_catching_data/saved_data")

RGB_SIZE = (1080, 1440)  # (w, h) after 90° rotate

EVENT_SHAPE = (1_000_000,)
EVENT_DTYPE = {
    "names": ["x", "y", "p", "t"],
    "formats": ["<u2", "<u2", "<i2", "<i8"],
    "offsets": [0, 2, 4, 8],
    "itemsize": 16,
}
FLIR_SHAPE = (RGB_SIZE[1], RGB_SIZE[0], 3)  # (H, W, C) = (1440,1080,3)
FLIR_DTYPE = np.uint8


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard listener
# ──────────────────────────────────────────────────────────────────────────────
def keyboard_listener(system_state: Value, is_paused: Value, stop_evt: Event):
    from pynput import keyboard as kb

    print("Keyboard controls:")
    print("  t - Start playback")
    print("  p - Pause/Resume")
    print("  q - Quit")

    def on_press(key):
        try:
            if key.char == 't':
                system_state.value = 't'
                print("[KEYBOARD] Starting playback...")
            elif key.char == 'p':
                is_paused.value = 1 - is_paused.value
                status = "Paused" if is_paused.value else "Resumed"
                print(f"[KEYBOARD] {status}")
            elif key.char == 'q':
                print("[KEYBOARD] Quitting...")
                stop_evt.set()
                return False  # exit listener
        except AttributeError:
            pass  # non-char key

    with kb.Listener(on_press=on_press) as listener:
        listener.join()


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (copied from collect_data.py)
# ──────────────────────────────────────────────────────────────────────────────
def show_streams(shared_event_meta, shared_flir_meta, stop_evt: Event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    ev_mem, ev_dtype_desc, ev_shape = shared_event_meta
    flir_mem, flir_dtype, flir_shape = shared_flir_meta

    ev_shm = shared_memory.SharedMemory(name=ev_mem.name)
    flir_shm = shared_memory.SharedMemory(name=flir_mem.name)

    ev_np = np.ndarray(ev_shape, dtype=np.dtype(ev_dtype_desc), buffer=ev_shm.buf)
    flir_np = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    gen = PeriodicFrameGenerationAlgorithm(sensor_width=1080, sensor_height=1440, fps=100, palette=ColorPalette.Dark)
    cd_frame = np.zeros((1440, 1090, 3), dtype=np.uint8)

    while not stop_evt.is_set():
        gen.generate_frame(ev_np, cd_frame, accumulation_time_us=1000)

        # Resize both frames to the same size for overlay
        target_size = (int(RGB_SIZE[0] * 0.6), int(RGB_SIZE[1] * 0.6))
        rgb_resized = cv2.resize(flir_np, target_size)

        # Resize event frame to match RGB (720x1280 -> 1080x1440 then resize)
        evt_resized_to_rgb = cv2.resize(cd_frame, (RGB_SIZE[0], RGB_SIZE[1]))
        evt_resized = cv2.resize(evt_resized_to_rgb, target_size)

        # Create mask: detect non-black pixels in event frame
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
            stop_evt.set(); break

    ev_shm.close(); flir_shm.close(); cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Playback synchronized RGB and event data")
    parser.add_argument("--index", type=int, default=0, help="Record index to playback")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Data directory path")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    set_start_method("spawn")

    # Verify files exist
    event_path = Path(args.data_dir) / "event" / f"{args.index}.raw"
    rgb_path = Path(args.data_dir) / "rgb" / f"{args.index}.mp4"

    if not event_path.exists():
        print(f"Error: Event file not found: {event_path}")
        sys.exit(1)
    if not rgb_path.exists():
        print(f"Error: RGB file not found: {rgb_path}")
        sys.exit(1)

    print(f"Playing back record {args.index}")
    print(f"  Event: {event_path}")
    print(f"  RGB: {rgb_path}")
    print(f"  Speed: {args.speed}x")

    # Shared memory
    event_mem = shared_memory.SharedMemory(create=True, size=np.dtype(EVENT_DTYPE).itemsize * EVENT_SHAPE[0])
    flir_mem = shared_memory.SharedMemory(create=True, size=np.prod(FLIR_SHAPE) * np.dtype(FLIR_DTYPE).itemsize)
    shared_event_meta = (event_mem, EVENT_DTYPE, EVENT_SHAPE)
    shared_flir_meta = (flir_mem, FLIR_DTYPE, FLIR_SHAPE)

    # Sync primitives (matching read_from_raw and read_from_video requirements)
    stop_evt = Event()
    system_state = Value('u', 'w')  # 'w' = waiting/loading, 't' = playing, 'e' = exit
    record_index = Value('i', args.index)
    start_ts = Value('d', 0.0)
    raw_count = Value('i', 0)
    num_event = Value('i', 0)
    event_data_version = Value('i', 0)
    flir_data_version = Value('i', 0)
    event_broadcasting = Value('u', 'r')  # r = ready, f = finished
    video_broadcasting = Value('u', 'r')
    is_paused = Value('i', 0)
    current_rgb_frame = Value('i', 0)
    current_event_frame = Value('i', 0)

    # Signal handler
    def shutdown(sig, frame):
        print("\n[MAIN] Shutting down...")
        system_state.value = 'e'
        stop_evt.set()

    signal.signal(signal.SIGINT, shutdown)

    # Start processes
    procs = [
        Process(target=keyboard_listener, args=(system_state, is_paused, stop_evt)),
        Process(target=read_from_raw, args=(
            shared_event_meta, event_data_version, system_state, num_event,
            raw_count, record_index, start_ts, stop_evt, event_broadcasting,
            args.data_dir, 0, args.speed, current_event_frame, is_paused
        )),
        Process(target=read_from_video, args=(
            shared_flir_meta, flir_data_version, system_state, raw_count,
            record_index, start_ts, stop_evt, video_broadcasting,
            args.data_dir, args.speed, current_rgb_frame, is_paused
        )),
        Process(target=show_streams, args=(shared_event_meta, shared_flir_meta, stop_evt)),
    ]

    for p in procs:
        p.start()

    # Wait for data to load
    print("[MAIN] Loading data...")
    time.sleep(2)  # Give processes time to initialize
    print("[MAIN] Data loaded. Press 't' to start playback.")

    # Monitor playback
    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
            # Check if both streams finished
            if event_broadcasting.value == 'f' and video_broadcasting.value == 'f':
                print("[MAIN] Playback finished")
                break
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")

    # Cleanup
    system_state.value = 'e'
    stop_evt.set()

    for p in procs:
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()

    event_mem.close()
    event_mem.unlink()
    flir_mem.close()
    flir_mem.unlink()
    print("[MAIN] Shutdown complete.")