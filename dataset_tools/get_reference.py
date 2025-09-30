import math
import cv2
import os
import sys
import torch
from multiprocessing import Process, Array, Queue, Manager, shared_memory, Value
import multiprocessing
import signal
import numpy as np
import time
import argparse


# Add project path for module imports
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

# Import custom modules
from data_input.read_from_raw import read_from_raw
from data_input.read_from_video import read_from_video



def starter(system_state, stop_event):
    time.sleep(3)
    print("Starting system...")
    system_state.value = 't'


# Global variables for signal handler
started_processes = []
stop_event = None


def signal_handler(sig, frame):
    print("\nSignal received, initiating safe shutdown...")
    global stop_event, started_processes
    if stop_event:
        stop_event.set()
    shutdown_processes(started_processes)
    sys.exit(0)


def keyboard(shared_value, system_state):
    from pynput import keyboard
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
            # shared_value.value = key.char if hasattr(key, 'char') else str(key)
            if key.char == 't':
                system_state.value = 't'
        except AttributeError:
            system_state.value = 'e'

    with keyboard.Listener(on_press=on_press) as listener:

        listener.join()

def save(shared_event_data, shared_flir_data, num_event, system_state, stop_event):

    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN
    root_dir='/media/xinchi/data/Event_RGB_catching_data/reference'
    
    # Ensure all directories exist
    os.makedirs(f"{root_dir}/rgb", exist_ok=True)
    os.makedirs(f"{root_dir}/event", exist_ok=True)
    os.makedirs(f"{root_dir}/overlay", exist_ok=True)
    
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=0.0,
                                                       palette=ColorPalette.Dark)

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    count = 0
    # time.sleep(1)
    while not stop_event.is_set():
        if system_state.value == 't':
            flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()
            event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
            event_data = event_data[:num_event.value]
            rgb_image = flir_data
            event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
            event_frame = np.ascontiguousarray(event_frame)
            event_frame_gen.generate_frame(event_data, event_frame, accumulation_time_us=1000)
            image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
            event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # Display only overlay - 仅显示叠加图像
            # 1. Create overlay image with better visualization
            rgb_brightened = cv2.convertScaleAbs(rgb_image, alpha=0.7, beta=30)
            overlay = cv2.addWeighted(rgb_brightened, 0.7, event_image, 0.3, 0)
            
            # 2. Add informative text overlay
            cv2.putText(overlay, f"Frame: {count}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(overlay, f"Events: {len(event_data)}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            cv2.putText(overlay, "Press 't' to save", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(overlay, "Press ESC to exit", (20, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 3. Scale for better display
            scale = 0.8
            width = int(overlay.shape[1] * scale)
            height = int(overlay.shape[0] * scale)
            new_size = (width, height)
            
            display_img = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)
            
            # Display only the overlay
            cv2.imshow('RGB + Event Overlay', display_img)
            key = cv2.waitKey(1) & 0xFF
            
            # Enhanced saving section - 增强保存部分
            # Save data with confirmation
            print(f"Saving frame {count}: RGB image + Event data ({len(event_data)} events)")
            cv2.imwrite(f"{root_dir}/rgb/" + str(count) + ".jpg", rgb_image)
            np.save(f"{root_dir}/event/" + str(count) + ".npy", event_data)
            
            # Optional: Save the overlay for reference
            cv2.imwrite(f"{root_dir}/overlay/" + str(count) + "_overlay.jpg", overlay)
            
            print(f"Frame {count} saved successfully!")

            count += 1


def shutdown_processes(processes, timeout=5):
    """
    安全地关闭进程列表

    Args:
        processes: 要关闭的进程列表
        timeout: 等待进程正常退出的超时时间（秒）
    """
    if not processes:
        print("No processes to shutdown.")
        return

    print(f"Shutting down {len(processes)} processes...")

    # 第一阶段：等待进程正常退出
    print("Phase 1: Waiting for graceful shutdown...")
    start_time = time.time()

    while processes and (time.time() - start_time) < timeout:
        alive_processes = [p for p in processes if p.is_alive()]
        if not alive_processes:
            print("All processes exited gracefully.")
            return

        print(f"Waiting for {len(alive_processes)} processes to exit...")
        time.sleep(0.5)

    # 第二阶段：强制终止仍在运行的进程
    alive_processes = [p for p in processes if p.is_alive()]
    if alive_processes:
        print(f"Phase 2: Force terminating {len(alive_processes)} remaining processes...")

        for process in alive_processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()
            except Exception as e:
                print(f"Error terminating process {process.pid}: {e}")

        # 等待terminate生效
        time.sleep(1)

        # 第三阶段：如果还有进程，使用kill
        still_alive = [p for p in alive_processes if p.is_alive()]
        if still_alive:
            print(f"Phase 3: Force killing {len(still_alive)} stubborn processes...")

            for process in still_alive:
                try:
                    print(f"Killing process {process.pid}...")
                    process.kill()
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # 最终检查
        time.sleep(0.5)
        final_alive = [p for p in processes if p.is_alive()]
        if final_alive:
            print(f"Warning: {len(final_alive)} processes still running after shutdown!")
        else:
            print("All processes successfully terminated.")

    print("Process shutdown complete.")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Event RGB Catching System')
    parser.add_argument('--record_index', type=int, default=11,
                        help='Initial record index (default: 40)')
    parser.add_argument('--test_mode', type=str, default="Save",
                        help='Test mode (default: Model). Options: Model, Event, RGB, Show')
    args = parser.parse_args()

    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    # Initialize stop_event for signal handler
    stop_event = multiprocessing.Event()
    started_processes.clear()  # Reset the list

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
    # ROI_memory=np.zeros(4)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_memory.buf)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    shared_ball_pixel_3D_data = (ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape)

    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True, size=np.prod(impact_point_shape) * np.dtype(
        impact_point_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    impact_point = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(3)
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

    print(f"Starting with record_index: {args.record_index}")

    record_dir = '/media/xinchi/data/Event_RGB_catching_data/saved_data_dc'

    # Define all available processes
    all_processes = {
        'event_process': Process(target=read_from_raw,
                                 args=(shared_event_data, system_state, num_event, raw_count, record_index, start_ts,
                                       stop_event, event_broadcasting, record_dir)),
        'flir_process': Process(target=read_from_video,
                                args=(shared_flir_data, system_state, raw_count, record_index, start_ts, stop_event,
                                      video_broadcasting, record_dir)),
        'keyboard_process': Process(target=keyboard, args=(keyboard_input, system_state)),
        'save_process': Process(target=save,
                                   args=(shared_event_data, shared_flir_data, num_event, system_state, stop_event)),
        'starter_process': Process(target=starter, args=(system_state, stop_event))
    }

    # Define process groups for different functionalities
    # YOU CAN MODIFY THESE GROUPS OR ADD NEW ONES
    process_groups = {
        'Save': ['event_process', 'flir_process', 'save_process', 'keyboard_process', 'starter_process'],
    }
    # Get processes to start for current mode
    processes_to_start = process_groups.get(args.test_mode, process_groups[args.test_mode])
    print(f"Starting mode: {args.test_mode}")
    print(f"Processes to start: {processes_to_start}")

    # Start selected processes dynamically
    # started_processes is already initialized as global variable
    for process_name in processes_to_start:
        if process_name in all_processes:
            process = all_processes[process_name]
            process.start()
            started_processes.append(process)
            print(f"Started: {process_name}")
        else:
            print(f"Warning: Process '{process_name}' not found!")
    # Wait for processes to finish or system_state becomes 'e'
    try:
        while True:
            # Check if system should exit
            if system_state.value == 'e':
                print("System exit requested, terminating processes...")
                stop_event.set()
                break

            # Check if all active processes are still alive
            if not any(p.is_alive() for p in started_processes):
                print("All processes finished naturally")
                break

            time.sleep(0.1)  # Check every 100ms

        # Safe shutdown of all started processes
        shutdown_processes(started_processes)

    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
        stop_event.set()
        # Safe shutdown of all started processes
        shutdown_processes(started_processes)
    finally:
        print("Cleaning up shared memory...")
        for shm in [event_memory, flir_memory, ROI_memory, ball_pixel_memory,
                    ball_pixel_3D_memory, impact_point_memory]:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # Already unlinked
                pass

