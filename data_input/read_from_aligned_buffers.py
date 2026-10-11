import cv2
import os
import tempfile
from multiprocessing import shared_memory
import numpy as np
import time

# Import configuration
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
from replay_options import positive_ratio


def align_to_rgb(event_data, rgb_width=None, rgb_height=None, event_width=None, event_height=None):
    if rgb_width is None or rgb_height is None or event_width is None or event_height is None:
        dims = Config.get_image_dimensions()
        rgb_width = dims['rgb_width'] if rgb_width is None else rgb_width
        rgb_height = dims['rgb_height'] if rgb_height is None else rgb_height
        event_width = dims['event_width'] if event_width is None else event_width
        event_height = dims['event_height'] if event_height is None else event_height

    alignment = Config.get_alignment()
    event_pixel = alignment['event_pixel_size']
    rgb_pixel = alignment['rgb_pixel_size']
    offset_x = alignment['offset_x']
    offset_y = alignment['offset_y']

    x_e, y_e = event_data['x'].copy(), event_data['y'].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2

    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + offset_x
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + offset_y
    y = np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)


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
    raw_path = os.path.join(record_dir, f"event/{index}.raw")
    dat_path = os.path.join(record_dir, f"event/{index}.dat")
    # Newly generated DAT takes precedence over a legacy same-index RAW.
    if os.path.isfile(dat_path):
        return dat_path
    if os.path.isfile(raw_path):
        return raw_path
    raise FileNotFoundError(f"No event file found for index {index} in {record_dir}")


def trim_buffers(event_buffer, rgb_buffer, ratio):
    ratio = positive_ratio(ratio)
    event_count = len(event_buffer)
    rgb_count = len(rgb_buffer)
    print("Before trim: RGB frames =", rgb_count, ", Event packets =", event_count)

    if event_count == 0 or rgb_count == 0:
        return event_buffer, rgb_buffer

    target_event_count = rgb_count * ratio
    # print(target_event_count)
    if target_event_count > event_count:
        rgb_bias = int(rgb_count - event_count / ratio)
        # print(rgb_bias)
        if rgb_bias > 0:
            rgb_buffer = rgb_buffer[:-rgb_bias]
    else:

        event_bias = int(event_count - target_event_count)
        print(event_bias)
        if event_bias > 0:
            event_buffer = event_buffer[:-event_bias]

    return event_buffer, rgb_buffer


def load_processed_event_file(event_path, data_reading):
    """Read already rotated events and align them to RGB; never rotate at replay.

    File extensions and sensor headers do not reliably indicate orientation:
    historical DAT files can have a .raw suffix and retain the original header.
    The caller must supply processed data from the offline preprocessing step.
    """
    if Config.get_alignment().get('replay_apply_rotation', False) is not False:
        raise ValueError(
            "Replay requires already rotated recordings and "
            "alignment.replay_apply_rotation: false. Preprocess raw data offline."
        )
    iterator, cleanup = _build_events_iterator(
        input_path=str(event_path),
        mode="delta_t",
        delta_t=data_reading['delta_t'],
        n_events=data_reading['n_events'],
    )
    if iterator is None:
        raise RuntimeError(f"Failed to open event file: {event_path}")
    event_buffer = []
    try:
        print(f"Replay: processed input, no rotation; header HxW: {iterator.get_size()}", flush=True)
        for ev in iterator:
            # Keep SDK-owned packets intact and retain polarity and timestamps.
            aligned = ev.copy()
            aligned['x'], aligned['y'] = align_to_rgb(aligned)
            event_buffer.append(aligned)
    finally:
        if cleanup:
            cleanup()
    return event_buffer


def _load_event_buffer(record_dir, record_index, data_reading):
    return load_processed_event_file(
        resolve_event_path(record_dir, record_index), data_reading
    )


def _load_rgb_buffer(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video: {video_path}")

    rgb_buffer = []
    try:
        ret, frame = cap.read()
        while ret:
            rgb_buffer.append(frame)
            ret, frame = cap.read()
    finally:
        cap.release()

    return rgb_buffer


def read_from_aligned_buffers(shared_event_data, shared_flir_data, event_data_version, flir_data_version,
                              system_state, num_event, raw_count, record_index, start_ts, stop_event,
                              event_broadcasting, video_broadcasting, record_dir='',
                              playback_speed=1.0, ratio=10,
                              current_event_frame=None, current_rgb_frame=None, is_paused=None,
                              event_frame_offset=0):
    # Validate before attaching shared memory or opening recordings.
    ratio = positive_ratio(ratio)
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    event_array = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    flir_array_view = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    event_count = 0
    initialized = False
    event_buffer = []
    rgb_buffer = []
    period_event = None

    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        if system_state.value == 'w' and not initialized:
            data_reading = Config.get_data_reading()
            video_path = os.path.join(record_dir, f"rgb/{record_index.value}.mp4")
            print(video_path)

            event_buffer = _load_event_buffer(record_dir, record_index.value, data_reading)
            rgb_buffer = _load_rgb_buffer(video_path)

            event_buffer, rgb_buffer = trim_buffers(event_buffer, rgb_buffer, ratio)
            raw_count.value = len(event_buffer)

            period_event = data_reading['send_interval'] / playback_speed
            print(f"Aligned buffers ready. Event packets: {len(event_buffer)}, RGB frames: {len(rgb_buffer)}")
            print(f"Event period: {period_event:.4f} seconds, ratio: {ratio}")

            event_count = 0
            initialized = True

        if system_state.value == 't' and initialized:
            if start_ts.value == 0.0:
                start_ts.value = time.perf_counter()

            if event_count >= len(event_buffer):
                event_broadcasting.value = 'f'
                video_broadcasting.value = 'f'
                initialized = False
                print("End of aligned data.")
                continue

            loop_idx = event_count
            target_time = start_ts.value + loop_idx * period_event

            adjusted_event_count = event_count + event_frame_offset
            adjusted_event_count = max(0, min(adjusted_event_count, len(event_buffer) - 1))

            ev = event_buffer[adjusted_event_count]
            n = ev.shape[0]
            num_event.value = n
            if n > 0:
                event_array[:min(n, 100000)] = ev[:min(n, 100000)]
                event_data_version.value += 1
                if event_count % 100 == 0:
                    print(f"Event frame {event_count}, version: {event_data_version.value}, events: {n}")

            if current_event_frame is not None:
                current_event_frame.value = adjusted_event_count

            if event_count % ratio == 0:
                rgb_index = event_count // ratio
                if rgb_index < len(rgb_buffer):
                    cv_image = rgb_buffer[rgb_index]
                    processing = Config.get_image_processing()
                    brightness_multiplier = processing['brightness_multiplier']
                    if brightness_multiplier != 1.0:
                        cv_image = np.clip(cv_image.astype(np.float32) * brightness_multiplier, 0, 255).astype(np.uint8)
                    flir_array_view[:] = cv_image
                    flir_data_version.value += 1
                    if current_rgb_frame is not None:
                        current_rgb_frame.value = rgb_index

            sleep_time = target_time + period_event - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            while is_paused and is_paused.value == 1 and not stop_event.is_set():
                time.sleep(0.1)

            event_count += 1
