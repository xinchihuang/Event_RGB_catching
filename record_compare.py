import cv2
import os
import sys
import torch
from multiprocessing import Process, shared_memory, Value
import multiprocessing
import signal
import numpy as np
import time
import argparse
from process_lifecycle import run_workers
from pathlib import Path
import subprocess

# Add project path for module imports
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


def resolve_config_file(task, test_mode):
    from config.config_loader import config_file_for
    mode = "Event" if (test_mode or "").lower() == "playback" else test_mode
    return config_file_for(task, mode)


def set_active_config(task, test_mode):
    from config.config_loader import initialize_config
    mode = "Event" if (test_mode or "").lower() == "playback" else test_mode
    return str(initialize_config(task, mode).config_path)


def save_tracking_results(mode, task, record_index, ball_pixels, count, output_root):
    mode_name = (mode or "").strip()
    if mode_name not in {"Model", "RGB", "Event"}:
        return None
    if count <= 0:
        print(f"No tracking results to save for {mode_name}.")
        return None
    base_dir = os.path.join(output_root, f"{mode_name}_{task}")
    save_dir = os.path.join(base_dir, "ball_pixel_3D_records_ball")
    os.makedirs(save_dir, exist_ok=True)
    data = ball_pixels[:count].copy()
    x = data[:, 0]
    y = data[:, 1]
    size = data[:, 2]
    t = data[:, 3]
    half = size / 2.0
    x1 = x - half
    y1 = y - half
    x2 = x + half
    y2 = y + half
    out = np.column_stack([x, y, size, t, x1, y1, x2, y2])
    filename = os.path.join(save_dir, f"{record_index}.csv")
    header = "center_x,center_y,size,t,x1,y1,x2,y2"
    np.savetxt(filename, out, delimiter=',', fmt='%.6f', header=header, comments='')
    print(f"Saved tracking results to {filename}")
    return filename

# Import custom modules
from data_input.read_from_aligned_buffers import read_from_aligned_buffers
# from data_input.read_from_files_simple import read_from_raw_files as read_from_raw
# from data_input.read_from_files_simple import read_from_video_files as read_from_video
from tracking.ROI_generation.AMTTrack_ROI import model
from RGB_baseline.RGB_ROI import pure_RGB
from RGB_baseline.HSV_tracker import RGB_tracking_3D
from tracking.Event_tracker import event_tracking_3D_direct_compare
from evaluation_tools.gui_process_compare import (
    gui_display_process_compare,
    gui_display_process_compare_from_files,
)

# Import configuration


def starter(system_state, stop_event):
    if stop_event.wait(3) or system_state.value == 'e':
        return
    print("Starting system...")
    system_state.value = 't'
    print("system started")

    # Remain alive until the parent requests shutdown.
    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        time.sleep(0.01)  # Small delay to avoid excessive CPU usage


# Shared stop request used by the parent signal handler
stop_event = None


def signal_handler(sig, frame):
    print("\nSignal received, initiating safe shutdown...")
    # The main loop performs shutdown and cleanup exactly once.
    if stop_event:
        stop_event.set()


def keyboard(shared_value, system_state):
    from pynput import keyboard
    import threading
    import time

    def play_voice_notification(message):
        """Play voice notification"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speech rate
            engine.setProperty('volume', 0.9)  # Volume level
            engine.say(message)
            engine.runAndWait()
        except ImportError:
            print(f"Voice notification: {message} (pyttsx3 not available)")
        except Exception as e:
            print(f"Voice playback failed: {e}, using text notification: {message}")

    def state_monitor():
        """Monitor state changes and trigger voice notifications"""
        previous_state = system_state.value
        while True:
            current_state = system_state.value
            if previous_state != current_state and current_state == 't':
                print(f"State change detected: {previous_state} -> {current_state}")
                # Play voice in new thread to avoid blocking
                voice_thread = threading.Thread(
                    target=play_voice_notification,
                    args=("System started tracking",),
                    daemon=True
                )
                voice_thread.start()
            previous_state = current_state
            time.sleep(0.1)  # Check interval 100ms

    # Start state monitoring thread
    monitor_thread = threading.Thread(target=state_monitor, daemon=True)
    monitor_thread.start()

    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
            # shared_value.value = key.char if hasattr(key, 'char') else str(key)
            if key.char == 't':
                system_state.value = 't'
            elif key.char == 'w':
                system_state.value = 'w'
        except AttributeError:
            system_state.value = 'e'

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def reseter_for_multi_record(shared_event_data, shared_flir_data, shared_ROI_data, shared_ball_pixel_3D_data, shared_ball_pixel_3D_data_raw, shared_ball_pixel_3D_data_rgb,
                             ball_observations,
                             shared_impact_point_data, raw_count, num_event, record_index, start_ts, event_broadcasting,
                             video_broadcasting, event_data_version, flir_data_version, system_state, stop_event):
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ball_pixel_3D_raw_memory, ball_pixel_3D_raw_dtype, ball_pixel_3D_raw_shape = shared_ball_pixel_3D_data_raw
    ball_pixel_3D_raw_shm = shared_memory.SharedMemory(name=ball_pixel_3D_raw_memory.name)
    ball_pixel_3D_rgb_memory, ball_pixel_3D_rgb_dtype, ball_pixel_3D_rgb_shape = shared_ball_pixel_3D_data_rgb
    ball_pixel_3D_rgb_shm = shared_memory.SharedMemory(name=ball_pixel_3D_rgb_memory.name)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)

    while not stop_event.is_set():

        if system_state.value == 'e':
            break
        # print(event_broadcasting.value,video_broadcasting.value)
        # print(event_broadcasting.value,video_broadcasting.value,system_state.value)
        # if event_broadcasting.value=='f' and video_broadcasting.value=='f':
        #     system_state.value='r'
        # print(event_broadcasting.value,video_broadcasting.value,system_state.value)
        if event_broadcasting.value == 'f' or video_broadcasting.value == 'f':
            system_state.value = 'r'
        if system_state.value == 'r':
            print("Waiting for reset")
            time.sleep(2)
            system_state.value = 'w'
            print("Set to Waiting")
            time.sleep(2)
            print("Cleaning memory...")
            # 清空 event memory
            event_array = np.ndarray(event_shape, dtype=np.dtype(event_dtype), buffer=event_shm.buf)
            event_array[:] = np.zeros(event_shape, dtype=np.dtype(event_dtype))

            # 清空 flir image
            flir_array = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
            flir_array[:] = 0

            # 清空 3D ball pixel
            ball_pixel_3D_array = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype,
                                             buffer=ball_pixel_3D_shm.buf)
            ball_pixel_3D_array[:] = 0
            
            # 清空 3D ball pixel raw
            ball_pixel_3D_raw_array = np.ndarray(ball_pixel_3D_raw_shape, dtype=ball_pixel_3D_raw_dtype,
                                                 buffer=ball_pixel_3D_raw_shm.buf)
            ball_pixel_3D_raw_array[:] = 0
            
            # 清空 3D ball pixel rgb
            ball_pixel_3D_rgb_array = np.ndarray(ball_pixel_3D_rgb_shape, dtype=ball_pixel_3D_rgb_dtype,
                                                 buffer=ball_pixel_3D_rgb_shm.buf)
            ball_pixel_3D_rgb_array[:] = 0

            # 清空 ROI
            ROI_array = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
            ROI_array[:] = 0

            # 清空 impact point
            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            impact_point_array[:] = 0

            # 重置 ball_observations
            ball_observations.value = 0

            raw_count.value = 0
            # 重置 num_event
            num_event.value = 0
            # 下一个record

            start_ts.value = 0.0
            video_broadcasting.value = 's'
            event_broadcasting.value = 's'

            # 重置数据版本计数器
            event_data_version.value = 0
            flir_data_version.value = 0

            print("Restarting system...")
            time.sleep(2)
            record_index.value += 1
            system_state.value = 'e'


def check_align(shared_event_data, shared_flir_data, num_event, event_data_version, system_state, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=0.0,
                                                       palette=ColorPalette.Dark)

    print(event_frame_gen.get_accumulation_time_us())
    print(event_frame_gen.get_fps())
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    dbscan = DBSCAN(eps=9, min_samples=70)
    count = 0
    buffered_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)

    # Track last processed data version to skip stale data
    last_event_version = -1

    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # Check if we have new data, skip if stale
        current_event_version = event_data_version.value
        if current_event_version == last_event_version:
            time.sleep(0.001)  # Brief sleep to avoid busy-waiting
            continue
        last_event_version = current_event_version

        # start_time = time.time()
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()
        event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
        if num_event.value == 0:
            continue
        event_data = event_data[:num_event.value]
        rgb_image = flir_data
        event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame, accumulation_time_us=1000)
        image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
        dbscan.fit(points)
        labels = dbscan.labels_
        event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cluster_points = points[labels != -1]
        if not len(cluster_points) == 0:
            y = min(cluster_points[:, 0])
            x = min(cluster_points[:, 1])
            w = max(cluster_points[:, 0]) - y
            h = max(cluster_points[:, 1]) - x
            box_thickness = 20
            cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), box_thickness)
            cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), box_thickness)
        # combined = np.hstack((rgb_image, event_image))
        rgb_image = cv2.convertScaleAbs(rgb_image, alpha=0.5, beta=30)
        combined = cv2.addWeighted(event_image, 0.3, rgb_image, 0.2, 0)
        scale = 0.8
        # 获取新的尺寸
        width = int(combined.shape[1] * scale)
        height = int(combined.shape[0] * scale)
        new_size = (width, height)

        resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

        cv2.imshow('Image with Bounding Box', resized_img)
        cv2.waitKey(1)
        # cv2.imwrite(f"{root_dir}/{0}/combined/" + str(count) + ".jpg", combined)
        # cv2.imwrite(f"{root_dir}/{index}/event/" + str(rgb_index) + ".jpg", event_image)

        count += 1


def model_tracking_writer(shared_event_data, shared_ROI_data, shared_ball_pixel_3D_data,
                          ball_observations, num_event, event_data_version, system_state, stop_event):
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    event_array_view = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI_array_view = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

    ball_memory, ball_dtype, ball_shape = shared_ball_pixel_3D_data
    ball_shm = shared_memory.SharedMemory(name=ball_memory.name)
    ball_array = np.ndarray(ball_shape, dtype=ball_dtype, buffer=ball_shm.buf)

    last_event_version = -1
    try:
        while not stop_event.is_set():
            if system_state.value != 't':
                time.sleep(0.001)
                continue

            current_event_version = event_data_version.value
            if current_event_version == last_event_version:
                time.sleep(0.001)
                continue
            last_event_version = current_event_version

            current_num_events = num_event.value
            if current_num_events <= 0 or current_num_events > event_shape[0]:
                continue

            event_data = event_array_view[:current_num_events]
            if event_data.shape[0] == 0:
                continue

            x_r, y_r, w_r, h_r = ROI_array_view.copy()
            if w_r <= 0 or h_r <= 0:
                continue

            t = event_data['t'][-1] / 1000000.0
            center_x = float(x_r + w_r / 2.0)
            center_y = float(y_r + h_r / 2.0)
            size = float(max(w_r, h_r))

            idx = ball_observations.value
            if idx < ball_shape[0] - 1:
                ball_array[idx][0] = center_x
                ball_array[idx][1] = center_y
                ball_array[idx][2] = size
                ball_array[idx][3] = t
                ball_observations.value = idx + 1
    finally:
        print("Model tracking writer: Cleaning up...")




if __name__ == '__main__':
    # Parse command line arguments
    from replay_options import positive_ratio
    from project_paths import config_paths, stage_path, resolve_path, require_recordings
    parser = argparse.ArgumentParser(
        description='Event RGB Catching System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--record_index', type=int, default=9,
                        help='Initial record/output index')
    parser.add_argument('--test_mode', type=str, default="Playback",
                        help='Test mode: All (sequential+playback), Save (sequential only), Model, RGB, Event, Playback')
    parser.add_argument('--task', type=str, default="clutter",
                        choices=["normal", "dark", "clutter", "dark_clutter"],
                        help='Task configuration to use')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                        help='Playback speed for aligned event/RGB streams (1.0 = real-time)')
    parser.add_argument('--output_root', '--output-root', type=str, default=None,
                        help='Output root for saved tracking CSVs')
    parser.add_argument('--ratio', type=positive_ratio, default=10,
                        help='Event packets per RGB frame (positive integer)')
    parser.add_argument('--record-dir', help='Exact processed recording directory')
    args = parser.parse_args()

    # Validate every planned mode before launching sequential subprocesses.
    from config.config_loader import ConfigLoader
    modes_to_check = (["Model", "RGB", "Event", "Playback"]
                      if args.test_mode.lower() == "all" else
                      ["Model", "RGB", "Event"] if args.test_mode.lower() == "save"
                      else [args.test_mode])
    try:
        for mode in modes_to_check:
            requested = (os.environ["ACTIVE_CONFIG_FILE"] if "ACTIVE_CONFIG_FILE" in os.environ
                         else resolve_config_file(args.task, mode))
            loaded_config = ConfigLoader(requested)
    except (OSError, ValueError) as exc:
        parser.error(f"Configuration error: {exc}")
    try:
        if args.record_dir is not None:
            os.environ['RECORD_DIR'] = str(resolve_path(args.record_dir))
        paths = config_paths(loaded_config)
        os.environ['RECORD_DIR'] = str(require_recordings(paths['record_dir']))
        args.output_root = str(stage_path('demo', explicit=args.output_root))
    except (OSError, ValueError) as exc:
        parser.error(f"Data path error: {exc}")
    print(f"Processed input: {os.environ['RECORD_DIR']}\nComparison output: {args.output_root}", flush=True)


    if (args.test_mode or "").strip().lower() in {"all", "save"}:
        modes = ["Model", "RGB", "Event"]
        for mode in modes:
            print(f"=== Sequential run: {mode} ===")
            env = os.environ.copy()
            env["ENABLE_GUI"] = "0"
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--record_index",
                str(args.record_index),
                "--test_mode",
                mode,
                "--task",
                args.task,
                "--playback_speed",
                str(args.playback_speed),
                "--ratio", str(args.ratio),
                "--output_root",
                args.output_root,
            ]
            result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
            if result.returncode != 0:
                print(f"Run failed for mode {mode} with code {result.returncode}")
                sys.exit(result.returncode)

        if (args.test_mode or "").strip().lower() == "save":
            sys.exit(0)

        print("=== Playback comparison ===")
        env = os.environ.copy()
        env["ENABLE_GUI"] = "1"
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--record_index",
            str(args.record_index),
            "--test_mode",
            "Playback",
            "--task",
            args.task,
            "--playback_speed",
            str(args.playback_speed),
            "--ratio", str(args.ratio),
            "--output_root",
            args.output_root,
        ]
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
        sys.exit(result.returncode)

    # Set environment variables for subprocess config loading
    config_file = set_active_config(args.task, args.test_mode)
    if config_file:
        print(f"使用任务配置 {args.task}: {config_file}")
    else:
        print(f"未知任务类型: {args.task}, 使用默认配置")

    # Set environment variables for GUI process to access test_mode and task
    os.environ['TEST_MODE'] = args.test_mode
    print(f"设置测试模式: {args.test_mode}, 任务: {args.task}")

    device = torch.device("cuda")

    # Check if GUI should be enabled
    USE_GUI = os.environ.get('ENABLE_GUI', '1') == '1'
    print(f"GUI Display: {'Enabled' if USE_GUI else 'Disabled'}")

    multiprocessing.set_start_method('spawn')

    # Initialize stop_event for signal handler
    stop_event = multiprocessing.Event()

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    # Shared objects
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape = (100000,)
    event_memory = shared_memory.SharedMemory(create=True, size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data = (event_memory, event_dtype, event_shape)

    flir_dtype = np.uint8
    flir_shape = (1440, 1080, 3)
    flir_memory = shared_memory.SharedMemory(create=True, size=np.prod(flir_shape) * np.dtype(flir_dtype).itemsize)
    shared_flir_data = (flir_memory, flir_dtype, flir_shape)
    # stop_event already initialized above

    ROI_dtype = np.uint16
    ROI_shape = (4,)
    ROI_memory = shared_memory.SharedMemory(create=True, size=np.prod(ROI_shape) * np.dtype(ROI_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_memory.buf)
    ROI[:] = np.zeros(4)
    shared_ROI_data = (ROI_memory, ROI_dtype, ROI_shape)

    empty_ROI_dtype = np.uint16
    empty_ROI_shape = (4,)
    empty_ROI_memory = shared_memory.SharedMemory(create=True, size=np.prod(empty_ROI_shape) * np.dtype(empty_ROI_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    empty_ROI = np.ndarray(empty_ROI_shape, dtype=empty_ROI_dtype, buffer=empty_ROI_memory.buf)
    empty_ROI[:] = np.zeros(4)
    empty_shared_ROI_data = (empty_ROI_memory, empty_ROI_dtype, empty_ROI_shape)

    ball_pixel_dtype = np.float64
    ball_pixel_shape = (100, 3)
    ball_pixel_memory = shared_memory.SharedMemory(create=True,
                                                   size=np.prod(ball_pixel_shape) * np.dtype(ball_pixel_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ball_pixel = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_memory.buf)
    ball_pixel[:] = np.zeros(ball_pixel_shape)
    shared_ball_pixel_data = (ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape)

    ball_pixel_3D_dtype = np.float64
    ball_pixel_3D_shape = (1000, 4)
    ball_pixel_3D_memory = shared_memory.SharedMemory(create=True,
                                                       size=np.prod(ball_pixel_3D_shape) * np.dtype(
                                                           ball_pixel_3D_dtype).itemsize)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_memory.buf)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    shared_ball_pixel_3D_data = (ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape)

    # Create separate shared memory for RGB tracking process
    ball_pixel_3D_raw_dtype = np.float64
    ball_pixel_3D_raw_shape = (1000, 4)
    ball_pixel_3D_raw_memory = shared_memory.SharedMemory(create=True,
                                                           size=np.prod(ball_pixel_3D_raw_shape) * np.dtype(
                                                               ball_pixel_3D_raw_dtype).itemsize)
    ball_pixel_3D_raw = np.ndarray(ball_pixel_3D_raw_shape, dtype=ball_pixel_3D_raw_dtype, buffer=ball_pixel_3D_raw_memory.buf)
    ball_pixel_3D_raw[:] = np.zeros(ball_pixel_3D_raw_shape)
    shared_ball_pixel_3D_data_raw = (ball_pixel_3D_raw_memory, ball_pixel_3D_raw_dtype, ball_pixel_3D_raw_shape)

    # Create separate shared memory for RGB tracking process (RGB specific)
    ball_pixel_3D_rgb_dtype = np.float64
    ball_pixel_3D_rgb_shape = (1000, 4)
    ball_pixel_3D_rgb_memory = shared_memory.SharedMemory(create=True,
                                                           size=np.prod(ball_pixel_3D_rgb_shape) * np.dtype(
                                                               ball_pixel_3D_rgb_dtype).itemsize)
    ball_pixel_3D_rgb = np.ndarray(ball_pixel_3D_rgb_shape, dtype=ball_pixel_3D_rgb_dtype, buffer=ball_pixel_3D_rgb_memory.buf)
    ball_pixel_3D_rgb[:] = np.zeros(ball_pixel_3D_rgb_shape)
    shared_ball_pixel_3D_data_rgb = (ball_pixel_3D_rgb_memory, ball_pixel_3D_rgb_dtype, ball_pixel_3D_rgb_shape)

    # Create impact point shared memory (for compatibility with reseter function)
    impact_point_dtype = np.float64
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True,
                                                    size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize)
    impact_point = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(impact_point_shape)
    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    keyboard_input = Value('u', 'e')
    system_state = Value('u', 'w')
    event_broadcasting = Value('u', 's')
    video_broadcasting = Value('u', 's')
    ball_observations = Value('i', 0)
    raw_count = Value('i', 0)
    num_event = Value('i', 0)
    record_index = Value('i', args.record_index)
    start_ts = Value('d', 0.0)

    # Data version counters to prevent stale data processing
    event_data_version = Value('i', 0)  # Incremented each time event data is updated
    flir_data_version = Value('i', 0)  # Incremented each time FLIR data is updated

    print(f"Starting with record_index: {args.record_index}")

    # Get record directory from config
    from config.config_loader import Config

    paths = Config.get_paths()
    record_dir = paths['record_dir']
    output_root = args.output_root
    print(f"Record directory: {record_dir}")

    # Define all available processes
    results_paths = {
        "model": os.path.join(output_root, f"Model_{args.task}", "ball_pixel_3D_records_ball", f"{record_index.value}.csv"),
        "rgb": os.path.join(output_root, f"RGB_{args.task}", "ball_pixel_3D_records_ball", f"{record_index.value}.csv"),
        "event": os.path.join(output_root, f"Event_{args.task}", "ball_pixel_3D_records_ball", f"{record_index.value}.csv"),
    }

    all_processes = {
        'aligned_process': Process(
            target=read_from_aligned_buffers,
            args=(shared_event_data, shared_flir_data, event_data_version, flir_data_version,
                  system_state, num_event, raw_count, record_index, start_ts, stop_event,
                  event_broadcasting, video_broadcasting, record_dir, args.playback_speed),
            kwargs={'ratio': args.ratio},
        ),
        'model_ROI_process': Process(target=model,
                                     args=(shared_event_data, shared_flir_data, shared_ROI_data, num_event,
                                           event_data_version, flir_data_version, system_state, stop_event)),
        'model_tracking_process': Process(target=model_tracking_writer,
                                          args=(shared_event_data, shared_ROI_data, shared_ball_pixel_3D_data,
                                                ball_observations, num_event, event_data_version, system_state, stop_event)),
        'RGB_ROI_process': Process(target=pure_RGB,
                                   args=(shared_event_data, shared_flir_data, shared_ROI_data, num_event,
                                         event_data_version, flir_data_version, system_state, stop_event)),
        'compare_event_tracking_process': Process(target=event_tracking_3D_direct_compare,
                                          args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                                shared_ball_pixel_3D_data, shared_ball_pixel_3D_data_raw, ball_observations, num_event,
                                                event_data_version, flir_data_version, system_state, stop_event)),
        'RGB_tracking_process': Process(target=RGB_tracking_3D,
                                        args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                              shared_ball_pixel_3D_data_rgb, ball_observations, num_event,
                                              event_data_version, flir_data_version, system_state, stop_event)),
        'gui_process': Process(target=gui_display_process_compare,
                               args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                     shared_ball_pixel_3D_data, shared_ball_pixel_3D_data_rgb, shared_ball_pixel_3D_data_raw, ball_observations, num_event, event_data_version, flir_data_version, system_state, stop_event)),
        'gui_playback_process': Process(target=gui_display_process_compare_from_files,
                               args=(shared_event_data, shared_flir_data, num_event,
                                     event_data_version, flir_data_version, system_state, stop_event, results_paths)),
        'starter_process': Process(target=starter, args=(system_state, stop_event)),
        'reseter_process': Process(target=reseter_for_multi_record,
                                  args=(shared_event_data, shared_flir_data, shared_ROI_data, shared_ball_pixel_3D_data, shared_ball_pixel_3D_data_raw,shared_ball_pixel_3D_data_rgb,
                                        ball_observations, shared_impact_point_data, raw_count, num_event, record_index, start_ts,
                                        event_broadcasting, video_broadcasting, event_data_version, flir_data_version, system_state, stop_event)),
    }

    # Define process groups for different functionalities
    # YOU CAN MODIFY THESE GROUPS OR ADD NEW ONES
    process_groups = {
        'Model': ['aligned_process', 'model_ROI_process', 'compare_event_tracking_process'],
        'Event': ['aligned_process', 'compare_event_tracking_process', 'starter_process'],
        'RGB': ['aligned_process', 'RGB_ROI_process', 'RGB_tracking_process', 'starter_process'],
        'Playback': ['aligned_process', 'gui_playback_process', 'starter_process'],
    }
    # Get processes to start for current mode
    mode_key_map = {"model": "Model", "rgb": "RGB", "event": "Event", "playback": "Playback"}
    mode_key = mode_key_map.get((args.test_mode or "").strip().lower(), args.test_mode)
    processes_to_start = process_groups.get(mode_key)
    print(f"Starting mode: {mode_key}")
    print(f"Processes to start: {processes_to_start}")
    if not processes_to_start:
        print(f"Unknown test_mode: {args.test_mode}")
        sys.exit(1)

    exit_code = 1
    try:
        selected = [(name, all_processes[name]) for name in processes_to_start
                    if name not in {'gui_process', 'gui_playback_process'} or USE_GUI]
        exit_code = run_workers(
            selected, stop_event, system_state,
            finished=lambda: (event_broadcasting.value == 'f'
                              and video_broadcasting.value == 'f'),
        )
        # Do not publish partial tracking results from a failed worker group.
        mode_map = {"model": "Model", "rgb": "RGB", "event": "Event"}
        mode_name = mode_map.get((args.test_mode or "").strip().lower())
        if exit_code == 0 and mode_name:
            buffer = ball_pixel_3D_rgb if mode_name == "RGB" else ball_pixel_3D
            count = min(ball_observations.value, buffer.shape[0])
            save_tracking_results(mode_name, args.task, record_index.value, buffer, count, output_root)
    finally:
        # This also runs if worker startup or CSV writing raises an exception.
        print("Cleaning up shared memory...")
        for shm in [event_memory, flir_memory, ROI_memory, empty_ROI_memory, ball_pixel_memory,
                    ball_pixel_3D_memory, ball_pixel_3D_raw_memory, ball_pixel_3D_rgb_memory, impact_point_memory]:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
    sys.exit(exit_code)
