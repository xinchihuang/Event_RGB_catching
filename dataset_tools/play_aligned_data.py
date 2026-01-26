#!/usr/bin/env python3
import argparse
import os
import tempfile

import cv2
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm

# ---- Edit defaults here ----
RECORD_DIR_DEFAULT = "/media/xinchi/data/processed_data/saved_data_dc"
IMAGE_DIMS = {
    "rgb_width": 1080,
    "rgb_height": 1440,
    "event_width": 720,
    "event_height": 1280,
}
ALIGNMENT = {
    "event_pixel_size": 3.45,
    "rgb_pixel_size": 4.86,
    "offset_x": 30,
    "offset_y": 29,
    "y_coordinate_offset": 1279,
}
DATA_READING = {
    "delta_t": 2000,
    "n_events": 1000000,
}
VISUALIZATION = {
    "accumulation_time_us": 1000,
    "display_scale": 0.8,
    "playback_wait_ms": 20,
}


def _is_dat_event_file(file_path):
    try:
        with open(file_path, "rb") as f:
            head = f.read(256)
        return b"Data file containing" in head
    except OSError:
        return False


def _build_events_iterator(input_path, **kwargs):
    if not _is_dat_event_file(input_path):
        return EventsIterator(input_path=input_path, **kwargs), None

    if input_path.endswith(".dat"):
        return EventsIterator(input_path=input_path, **kwargs), None

    tmp_dir = tempfile.mkdtemp(prefix="event_dat_")
    link_path = os.path.join(tmp_dir, os.path.basename(input_path) + ".dat")
    try:
        os.symlink(input_path, link_path)
        iterator = EventsIterator(input_path=link_path, **kwargs)
    except Exception as e:
        print(f"DAT symlink failed: {e}")
        try:
            if os.path.exists(link_path):
                os.unlink(link_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass
        return None, None

    def _cleanup():
        try:
            os.unlink(link_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return iterator, _cleanup


def align_to_rgb(event_data, rgb_width=None, rgb_height=None, event_width=None, event_height=None):
    if rgb_width is None or rgb_height is None or event_width is None or event_height is None:
        rgb_width = IMAGE_DIMS["rgb_width"] if rgb_width is None else rgb_width
        rgb_height = IMAGE_DIMS["rgb_height"] if rgb_height is None else rgb_height
        event_width = IMAGE_DIMS["event_width"] if event_width is None else event_width
        event_height = IMAGE_DIMS["event_height"] if event_height is None else event_height

    event_pixel = ALIGNMENT["event_pixel_size"]
    rgb_pixel = ALIGNMENT["rgb_pixel_size"]
    offset_x = ALIGNMENT["offset_x"]
    offset_y = ALIGNMENT["offset_y"]

    x_e, y_e = event_data["x"].copy(), event_data["y"].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2

    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + offset_x
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + offset_y
    y = np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)


def resolve_event_path(record_dir, index):
    raw_path = os.path.join(record_dir, f"event/{index}.raw")
    dat_path = os.path.join(record_dir, f"event/{index}.dat")
    if os.path.exists(raw_path):
        return raw_path
    if os.path.exists(dat_path):
        return dat_path
    raise FileNotFoundError(f"No event file found for index {index} in {record_dir}")


def load_event_buffer(event_path, delta_t, ratio, n_events):
    read_delta_t = delta_t * ratio
    iterator, cleanup = _build_events_iterator(
        input_path=event_path,
        mode="delta_t",
        delta_t=read_delta_t,
        n_events=n_events,
    )
    if iterator is None:
        raise RuntimeError(f"Failed to build EventsIterator for {event_path}")

    event_buffer = []
    try:
        for ev in iter(iterator):
            if ev is None:
                continue
            if ev.size > 0:
                ev["x"], ev["y"] = ev["y"].copy(), -ev["x"].copy()
                ev["y"] += ALIGNMENT["y_coordinate_offset"]
            event_buffer.append(ev.copy())
    finally:
        if cleanup:
            cleanup()

    return event_buffer


def load_rgb_buffer(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    rgb_buffer = []
    try:
        ok, frame = cap.read()
        while ok:
            rgb_buffer.append(frame.copy())
            ok, frame = cap.read()
    finally:
        cap.release()

    return rgb_buffer


def trim_to_match(event_buffer, rgb_buffer):
    count = min(len(event_buffer), len(rgb_buffer))
    return event_buffer[:count], rgb_buffer[:count]


def apply_spatial_align_buffer(event_buffer):
    aligned = []
    for ev in event_buffer:
        if ev is None:
            aligned.append(ev)
            continue
        ev_local = ev.copy()
        if ev_local.size > 0:
            ev_local["x"], ev_local["y"] = align_to_rgb(ev_local)
        aligned.append(ev_local)
    return aligned


def visualize_alignment(rgb_buffer, event_buffer, apply_spatial_align=True):
    if not rgb_buffer or not event_buffer:
        print("No data to visualize.")
        return

    acc_time = VISUALIZATION["accumulation_time_us"]
    scale = VISUALIZATION["display_scale"]
    wait_ms = VISUALIZATION["playback_wait_ms"]

    rgb_h, rgb_w = rgb_buffer[0].shape[:2]
    if apply_spatial_align:
        event_w, event_h = rgb_w, rgb_h
    else:
        event_w, event_h = IMAGE_DIMS["event_width"], IMAGE_DIMS["event_height"]

    frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=event_w,
        sensor_height=event_h,
        fps=100,
        palette=ColorPalette.Dark,
    )

    event_frame_native = np.zeros((event_h, event_w, 3), dtype=np.uint8)
    paused = False
    step_mode = False
    frame_idx = 0

    print("Press 'q' to quit, 'p' to pause/resume, 's' to step frame by frame")
    while frame_idx < len(rgb_buffer):
        rgb_img = rgb_buffer[frame_idx]
        ev = event_buffer[frame_idx] if frame_idx < len(event_buffer) else None

        event_frame_native[:] = 0
        if ev is not None and ev.size > 0:
            ev_local = ev.copy()
            ev_local["x"] = np.clip(ev_local["x"], 0, event_w - 1)
            ev_local["y"] = np.clip(ev_local["y"], 0, event_h - 1)
            frame_gen.generate_frame(ev_local, event_frame_native, accumulation_time_us=acc_time)

        event_frame = event_frame_native
        if event_frame.shape[0] != rgb_h or event_frame.shape[1] != rgb_w:
            event_frame = cv2.resize(event_frame, (rgb_w, rgb_h))

        display_img = np.hstack((rgb_img, event_frame))
        if scale != 1.0:
            display_img = cv2.resize(
                display_img,
                (int(display_img.shape[1] * scale), int(display_img.shape[0] * scale)),
            )

        cv2.imshow("Aligned RGB | Event", display_img)
        if paused or step_mode:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or key == 27:
            break
        if key == ord("p"):
            paused = not paused
            step_mode = False
        elif key == ord("s"):
            step_mode = not step_mode
            paused = False

        if not paused:
            frame_idx += 1

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Play aligned event/RGB data.")
    parser.add_argument("--record-dir", default=RECORD_DIR_DEFAULT)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--ratio", type=int, default=10)
    parser.add_argument("--delta-t", type=int, default=DATA_READING["delta_t"])
    parser.add_argument("--n-events", type=int, default=DATA_READING["n_events"])
    parser.add_argument("--no-spatial-align", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    event_path = resolve_event_path(args.record_dir, args.index)
    video_path = os.path.join(args.record_dir, f"rgb/{args.index}.mp4")

    print(f"Loading events: {event_path}")
    event_buffer = load_event_buffer(
        event_path,
        delta_t=args.delta_t,
        ratio=args.ratio,
        n_events=args.n_events,
    )
    print(f"Event packets: {len(event_buffer)}")

    print(f"Loading video: {video_path}")
    rgb_buffer = load_rgb_buffer(video_path)
    print(f"RGB frames: {len(rgb_buffer)}")

    event_buffer, rgb_buffer = trim_to_match(event_buffer, rgb_buffer)
    print(f"After trim: RGB={len(rgb_buffer)}, Events={len(event_buffer)}")

    display_events = event_buffer
    if not args.no_spatial_align:
        display_events = apply_spatial_align_buffer(event_buffer)

    visualize_alignment(
        rgb_buffer,
        display_events,
        apply_spatial_align=not args.no_spatial_align,
    )


if __name__ == "__main__":
    main()
