#!/usr/bin/env python3
"""
Count event packets and RGB frames for saved recordings.

Usage:
  python3 dataset_tools/count_saved_data.py --record-dir ... --all
"""
import argparse
import os
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import SCENES, stage_path, resolve_path, require_recordings, distinct_output

import cv2

DEFAULT_DELTA_T = 2000
DEFAULT_N_EVENTS = 10000


def _is_dat_event_file(file_path):
    try:
        with open(file_path, "rb") as f:
            head = f.read(256)
        return b"Data file containing" in head
    except OSError:
        return False


def _build_events_iterator(input_path, **kwargs):
    from metavision_core.event_io import EventsIterator

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
    return None


def resolve_rgb_path(record_dir, index):
    rgb_path = os.path.join(record_dir, f"rgb/{index}.mp4")
    if os.path.exists(rgb_path):
        return rgb_path
    return None


def _parse_indices(dir_path, exts):
    if not os.path.isdir(dir_path):
        return set()
    indices = set()
    for name in os.listdir(dir_path):
        base, ext = os.path.splitext(name)
        if ext.lower() not in exts:
            continue
        if not base.isdigit():
            continue
        indices.add(int(base))
    return indices


def list_indices(record_dir):
    event_dir = os.path.join(record_dir, "event")
    rgb_dir = os.path.join(record_dir, "rgb")
    event_indices = _parse_indices(event_dir, {".raw", ".dat"})
    rgb_indices = _parse_indices(rgb_dir, {".mp4"})
    return sorted(event_indices | rgb_indices)


def count_event_packets(event_path, delta_t, n_events):
    iterator, cleanup = _build_events_iterator(
        input_path=event_path,
        mode="delta_t",
        delta_t=delta_t,
        n_events=n_events,
    )
    if iterator is None:
        raise RuntimeError(f"Failed to build EventsIterator for {event_path}")

    count = 0
    try:
        for ev in iter(iterator):
            if ev is None:
                continue
            count += 1
    finally:
        if cleanup:
            cleanup()
    return count


def count_rgb_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count > 0:
        cap.release()
        return count

    count = 0
    try:
        ok, _ = cap.read()
        while ok:
            count += 1
            ok, _ = cap.read()
    finally:
        cap.release()
    return count


def run_index(args, index):
    event_path = resolve_event_path(args.record_dir, index)
    rgb_path = resolve_rgb_path(args.record_dir, index)

    if not event_path or not rgb_path:
        print(f"Missing input for index {index}: {event_path} or {rgb_path}")
        return False, 0, 0

    event_count = count_event_packets(event_path, args.delta_t, args.n_events)
    rgb_count = count_rgb_frames(rgb_path)

    print(
        f"Index {index}: event packets = {event_count}, rgb frames = {rgb_count}"
    )
    return True, event_count, rgb_count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Count event packets and RGB frames.")
    parser.add_argument("--task", choices=SCENES, default="normal")
    parser.add_argument("--record-dir")
    parser.add_argument("--index", type=int, default=None,
                        help="Starting recording ID (default: 0); not used with --all")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=None,
                           help="Number of consecutive IDs to check (default: 1); skip missing pairs")
    selection.add_argument("--auto", action="store_true",
                           help="Count from --index until the first missing pair")
    selection.add_argument("--all", action="store_true",
                           help="Scan all IDs found in event/ or rgb/; report and skip missing pairs")
    parser.add_argument("--delta-t", type=int, default=DEFAULT_DELTA_T)
    parser.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS)
    args = parser.parse_args(argv)
    if args.all and args.index is not None:
        parser.error('--all cannot be combined with --index')
    if args.index is not None and args.index < 0:
        parser.error('--index must be non-negative')
    if args.count is not None and args.count <= 0:
        parser.error('--count must be positive')
    try:
        args.record_dir = str(stage_path("replay", args.task, args.record_dir))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.index = 0 if args.index is None else args.index
    args.count = 1 if args.count is None else args.count
    return args


def main():
    args = parse_args()
    require_recordings(args.record_dir)
    print(f"Counting input: {args.record_dir}", flush=True)
    total_events = 0
    total_frames = 0
    total_indices = 0

    if args.all:
        indices = list_indices(args.record_dir)
        for index in indices:
            ok, ev_count, rgb_count = run_index(args, index)
            if ok:
                total_events += ev_count
                total_frames += rgb_count
                total_indices += 1
    elif args.auto:
        index = args.index
        while True:
            ok, ev_count, rgb_count = run_index(args, index)
            if not ok:
                break
            total_events += ev_count
            total_frames += rgb_count
            total_indices += 1
            index += 1
    else:
        for offset in range(args.count):
            index = args.index + offset
            ok, ev_count, rgb_count = run_index(args, index)
            if not ok:
                continue
            total_events += ev_count
            total_frames += rgb_count
            total_indices += 1

    if total_indices > 1:
        print(
            f"Total indices = {total_indices}, total event packets = {total_events}, "
            f"total rgb frames = {total_frames}"

        )


if __name__ == "__main__":
    main()
