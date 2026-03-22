#!/usr/bin/env python3
"""
Export aligned event packs, RGB frames, and overlay previews for a record.

Usage:
  python3 dataset_tools/get_reference_rewrite.py --record_index N --record_dir ... --output_dir ...
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
from data_input.read_from_dat import align_to_rgb

mode='d'
REFERENCE_ROOT_DIR = os.environ.get("REFERENCE_ROOT_DIR", f"/media/xinchi/data/processed_data/reference_{mode}")
RECORD_DATA_DIR = os.environ.get(
    "RECORD_DATA_DIR",
    f"/media/xinchi/data/processed_data/saved_data_{mode}",
)
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
    except Exception as exc:
        print(f"DAT symlink failed: {exc}")
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


def resolve_event_path(record_dir, index):
    dat_path = os.path.join(record_dir, f"event/{index}.dat")
    raw_path = os.path.join(record_dir, f"event/{index}.raw")
    if os.path.exists(dat_path):
        return dat_path
    if os.path.exists(raw_path):
        return raw_path
    raise FileNotFoundError(f"No event file found for index {index} in {record_dir}")

def load_event_buffer(event_path, delta_t, n_events):


    iterator, cleanup = _build_events_iterator(
        input_path=event_path,
        mode="delta_t",
        delta_t=delta_t,
        n_events=n_events,
    )
    if iterator is None:
        raise RuntimeError(f"Failed to build EventsIterator for {event_path}")
    event_buffer = []
    try:
        for ev in iter(iterator):
            if ev is None:
                continue

            ev['x'], ev['y'] = align_to_rgb(ev)
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


def build_overlay(event_pack, rgb_image, frame_gen, rgb_width, rgb_height, accumulation_time_us):
    event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    event_frame = np.ascontiguousarray(event_frame)
    aligned_pack = event_pack
    if event_pack is not None and event_pack.size > 0:
        aligned_pack = event_pack.copy()
        frame_gen.generate_frame(aligned_pack, event_frame, accumulation_time_us=accumulation_time_us)

    image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    rgb_brightened = cv2.convertScaleAbs(rgb_image, alpha=0.7, beta=30)
    overlay = cv2.addWeighted(rgb_brightened, 0.7, event_image, 0.3, 0)
    return overlay, aligned_pack


def main():
    parser = argparse.ArgumentParser(description="Save aligned event packs and RGB frames with overlays.")
    parser.add_argument("--record_dir", default=RECORD_DATA_DIR, help="Dataset root containing event/ and rgb/")
    parser.add_argument("--record_index", type=int, default=0, help="Record index to load")
    parser.add_argument("--event_path", default="", help="Override event file path (.dat/.raw)")
    parser.add_argument("--video_path", default="", help="Override rgb video path (.mp4)")
    parser.add_argument("--output_dir", default=REFERENCE_ROOT_DIR, help="Output root directory")
    parser.add_argument("--ratio", type=int, default=10, help="Event-to-RGB ratio (e.g. 10 => 10 event packs per RGB frame)")
    parser.add_argument("--delta_t", type=int, default=None, help="Event packet delta_t (us)")
    parser.add_argument("--n_events", type=int, default=None, help="Event packet max events")
    parser.add_argument("--accumulation_time_us", type=int, default=1000, help="Event frame accumulation time")
    parser.add_argument("--display", action="store_true", help="Show overlay preview while saving")
    parser.add_argument("--display_scale", type=float, default=None, help="Overlay display scale")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit number of frames (0 = all)")
    args = parser.parse_args()

    dims = Config.get_image_dimensions()
    alignment = Config.get_alignment()
    data_reading = Config.get_data_reading()

    delta_t = data_reading["delta_t"] if args.delta_t is None else args.delta_t
    n_events = data_reading["n_events"] if args.n_events is None else args.n_events
    image_processing = Config.get_image_processing()
    display_scale = image_processing["display_scale"] if args.display_scale is None else args.display_scale

    event_path = args.event_path or resolve_event_path(args.record_dir, args.record_index)
    video_path = args.video_path or os.path.join(args.record_dir, f"rgb/{args.record_index}.mp4")

    print(f"Event path: {event_path}")
    print(f"Video path: {video_path}")

    event_buffer = load_event_buffer(event_path, delta_t, n_events)
    print(event_buffer)
    rgb_buffer = load_rgb_buffer(video_path)

    event_count = len(event_buffer)
    rgb_count = len(rgb_buffer)
    print(f"Loaded {event_count} event packets, {rgb_count} RGB frames")

    max_frames_by_event = (event_count + args.ratio - 1) // args.ratio if event_count > 0 else 0
    frame_count = min(rgb_count, max_frames_by_event)
    if args.max_frames and args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if event_count != rgb_count * args.ratio:
        print(f"Warning: event packets != rgb_count * ratio, using {frame_count}")

    rgb_dir = os.path.join(args.output_dir, "rgb")
    event_dir = os.path.join(args.output_dir, "event")
    overlay_dir = os.path.join(args.output_dir, "overlay")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    rgb_width = dims["rgb_width"]
    rgb_height = dims["rgb_height"]
    frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=rgb_width,
        sensor_height=rgb_height,
        fps=0.0,
        palette=ColorPalette.Dark,
    )

    for idx in range(frame_count):
        rgb_image = rgb_buffer[idx]
        event_pack =event_buffer[idx*args.ratio]
        overlay, aligned_pack = build_overlay(
            event_pack,
            rgb_image,
            frame_gen,
            rgb_width,
            rgb_height,
            args.accumulation_time_us,
        )

        if idx == 0:
            if aligned_pack is None or aligned_pack.size == 0:
                print("First event pack is empty")
            else:
                print(f"First event pack size: {aligned_pack.shape[0]}")
                print(f"x range: {aligned_pack['x'].min()}..{aligned_pack['x'].max()}")
                print(f"y range: {aligned_pack['y'].min()}..{aligned_pack['y'].max()}")

        np.save(os.path.join(event_dir, f"{idx}.npy"), aligned_pack)
        cv2.imwrite(os.path.join(rgb_dir, f"{idx}.jpg"), rgb_image)
        cv2.imwrite(os.path.join(overlay_dir, f"{idx}_overlay.jpg"), overlay)

        if args.display:
            display = overlay
            if display_scale and display_scale != 1.0:
                width = int(display.shape[1] * display_scale)
                height = int(display.shape[0] * display_scale)
                display = cv2.resize(display, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow("RGB + Event Overlay", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if idx % 50 == 0:
            print(f"Saved frame {idx}/{frame_count}")

    if args.display:
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
