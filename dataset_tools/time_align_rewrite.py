#!/usr/bin/env python3
import argparse
import os
import tempfile

import cv2
import numpy as np
from metavision_core.event_io import EventsIterator, DatWriter
from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm
# ---- Edit defaults here ----
RECORD_DIR_DEFAULT = "/media/xinchi/data/saved_data_raw/saved_data_l"
SAVE_DIR_DEFAULT = "/media/xinchi/data/processed_data/saved_data_l"
# RECORD_DIR_DEFAULT = "/media/xinchi/data/processed_data/saved_data_dc"
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
    "n_events": 10000,
    "video_period": 0.02,
}
VISUALIZATION = {
    "accumulation_time_us": 1000,
    "display_scale": 0.8,
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


def load_event_buffer(event_path, delta_t, n_events):
    delta_t = DATA_READING["delta_t"] if delta_t is None else delta_t
    n_events = DATA_READING["n_events"] if n_events is None else n_events

    iterator, cleanup = _build_events_iterator(
        input_path=event_path,
        mode="delta_t",
        delta_t=delta_t,
        n_events=n_events,
    )
    if iterator is None:
        raise RuntimeError(f"Failed to build EventsIterator for {event_path}")

    event_size = iterator.get_size()
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

    return event_buffer, event_size


def load_rgb_buffer(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    rgb_buffer = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        ok, frame = cap.read()
        while ok:
            rgb_buffer.append(frame.copy())
            ok, frame = cap.read()
    finally:
        cap.release()

    return rgb_buffer, fps


def trim_buffers(event_buffer, rgb_buffer, ratio):
    event_count = len(event_buffer)
    rgb_count = len(rgb_buffer)
    print("Before trim: RGB frames =", rgb_count, ", Event packets =", event_count)

    if event_count == 0 or rgb_count == 0:
        return event_buffer, rgb_buffer

    target_event_count = rgb_count * ratio
    if target_event_count > event_count:
        rgb_bias = int(rgb_count - event_count / ratio)
        if rgb_bias > 0:
            rgb_buffer = rgb_buffer[:-rgb_bias]
    else:
        event_bias = int(event_count - target_event_count)
        if event_bias > 0:
            event_buffer = event_buffer[event_bias:]

    return event_buffer, rgb_buffer


def build_aligned_event_buffer(event_buffer, rgb_count, ratio):
    if rgb_count == 0:
        return []
    aligned = []
    max_idx = len(event_buffer) - 1
    for i in range(rgb_count):
        ev_idx = min(i * ratio, max_idx)
        aligned.append(event_buffer[ev_idx])
    return aligned


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


def save_event_buffer(event_buffer, output_path, event_size=None):
    if not event_buffer:
        print("No event data to save.")
        return False

    if event_size and event_size[0] and event_size[1]:
        height, width = int(event_size[0]), int(event_size[1])
    else:
        height, width = IMAGE_DIMS["event_height"], IMAGE_DIMS["event_width"]
    print(event_size)
    writer = DatWriter(output_path, height=height, width=width)
    try:
        for ev in event_buffer:
            writer.write(ev)
    finally:
        writer.close()

    print(f"Saved {len(event_buffer)} event packets to {output_path}")
    return True


def save_rgb_buffer(rgb_buffer, output_path, fps):
    if not rgb_buffer:
        print("No RGB frames to save.")
        return False

    h, w = rgb_buffer[0].shape[:2]
    if fps is None or fps <= 0:
        fps = 1.0 / DATA_READING["video_period"] if DATA_READING["video_period"] > 0 else 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in rgb_buffer:
        writer.write(frame)
    writer.release()

    print(f"Saved {len(rgb_buffer)} frames to {output_path}")
    return True


def visualize_alignment(rgb_buffer, event_buffer, apply_spatial_align=True):
    if not rgb_buffer or not event_buffer:
        print("No data to visualize.")
        return

    acc_time = VISUALIZATION["accumulation_time_us"]
    scale = VISUALIZATION["display_scale"]

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
            key = cv2.waitKey(20) & 0xFF

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
    parser = argparse.ArgumentParser(description="Load and align event/RGB buffers.")
    parser.add_argument("--record-dir", default=RECORD_DIR_DEFAULT)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--ratio", type=int, default=10)
    parser.add_argument("--delta-t", type=int, default=DATA_READING["delta_t"])
    parser.add_argument("--n-events", type=int, default=DATA_READING["n_events"])
    parser.add_argument("--no-spatial-align", action="store_true")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--save-dir", default=SAVE_DIR_DEFAULT)
    return parser.parse_args()


def run_index(args, index):
    event_path = os.path.join(args.record_dir, f"event/{index}.raw")
    video_path = os.path.join(args.record_dir, f"rgb/{index}.mp4")

    if not os.path.exists(event_path) or not os.path.exists(video_path):
        print(f"Missing input for index {index}: {event_path} or {video_path}")
        return False

    print(f"Loading events: {event_path}")
    event_buffer, event_size = load_event_buffer(
        event_path,
        delta_t=args.delta_t,
        n_events=args.n_events,
    )
    print(event_size)
    print(f"Event packets: {len(event_buffer)}")

    print(f"Loading video: {video_path}")
    rgb_buffer, rgb_fps = load_rgb_buffer(video_path)
    print(f"RGB frames: {len(rgb_buffer)}")

    event_buffer, rgb_buffer = trim_buffers(event_buffer, rgb_buffer, args.ratio)
    print(f"After trim: RGB={len(rgb_buffer)}, Events={len(event_buffer)}")

    aligned_event_buffer = build_aligned_event_buffer(event_buffer, len(rgb_buffer), args.ratio)
    print(f"Aligned pairs (for display): {len(aligned_event_buffer)}")

    save_dir = args.save_dir or f"{args.record_dir}_aligned"
    event_out_dir = os.path.join(save_dir, "event")
    rgb_out_dir = os.path.join(save_dir, "rgb")
    os.makedirs(event_out_dir, exist_ok=True)
    os.makedirs(rgb_out_dir, exist_ok=True)

    event_out_path = os.path.join(event_out_dir, f"{index}.raw")
    rgb_out_path = os.path.join(rgb_out_dir, f"{index}.mp4")
    # Save the trimmed event buffer; aligned buffer is only for visualization.
    save_event_buffer(event_buffer, event_out_path, event_size=event_size)
    save_rgb_buffer(rgb_buffer, rgb_out_path, rgb_fps)

    if not args.no_view:
        display_events = aligned_event_buffer
        if not args.no_spatial_align:
            display_events = apply_spatial_align_buffer(aligned_event_buffer)
        visualize_alignment(
            rgb_buffer,
            display_events,
            apply_spatial_align=not args.no_spatial_align,
        )

    return True


def main():
    args = parse_args()

    if args.auto:
        index = args.index
        while run_index(args, index):
            index += 1
    else:
        for offset in range(args.count):
            index = args.index + offset
            if not run_index(args, index):
                break


if __name__ == "__main__":
    main()
