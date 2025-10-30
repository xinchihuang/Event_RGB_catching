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
from gi.overrides.Gtk import initialized

# Add project path for module imports
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

# Import custom modules
from data_input.read_from_raw import read_from_raw
from data_input.read_from_video import read_from_video
from ROI_generation.AMTTrack_ROI import model
from ROI_generation.RGB_ROI import pure_RGB
from tracking_2D.HSV_tracker import RGB_tracking_3D
from tracking_2D.Event_tracker import event_tracking_3D_direct as event_tracking_3D
from fitting.fitting_process import landing_prediction
from hardware.motor import robot_control
from gui_process import gui_display_process

# Import configuration
from config.config_loader import Config

def starter(system_state,stop_event):
    system_state.value='w'
    time.sleep(3)
    print("Starting system...")
    system_state.value = 't'
    print("system started")
    
    # Keep system_state = 't' alive like model_ROI_process does
    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        system_state.value = 't'  # Continuously maintain tracking state
        time.sleep(0.01)  # Small delay to avoid excessive CPU usage

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
def keyboard(shared_value,system_state):
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
            if key.char=='t':
                system_state.value='t'
            elif key.char=='w':
                system_state.value='w'
        except AttributeError:
            system_state.value='e'

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
def reseter_for_multi_record(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,
                             shared_impact_point_data,raw_count,num_event,record_index,start_ts,event_broadcasting,video_broadcasting,event_data_version,flir_data_version,system_state,stop_event):
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
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
        if event_broadcasting.value=='f' or video_broadcasting.value=='f':
            system_state.value='r'
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

            # 清空 ROI
            ROI_array = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
            ROI_array[:] = 0

            # 清空 impact point
            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            impact_point_array[:] = 0

            # 重置 ball_observations
            ball_observations.value = 0

            raw_count.value=0
            # 重置 num_event
            num_event.value = 0
            # 下一个record

            start_ts.value = 0.0
            video_broadcasting.value='s'
            event_broadcasting.value='s'

            # 重置数据版本计数器
            event_data_version.value = 0
            flir_data_version.value = 0

            print("Restarting system...")
            time.sleep(2)
            record_index.value += 1
            system_state.value = 'e'

def check_align(shared_event_data,shared_flir_data,num_event,event_data_version,system_state,stop_event):
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
    count=0
    # time.sleep(1)
    # root_dir = "/home/xinchi/Event_RGB_catching_data/dataset"
    # index=len([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    # if not os.path.exists(f"{root_dir}/{index}"):
    #     os.makedirs(f"{root_dir}/{index}")
    #     os.makedirs(f"{root_dir}/{index}/combined")
    #     os.makedirs(f"{root_dir}/{index}/rgb")
    #     os.makedirs(f"{root_dir}/{index}/event")
    # time.sleep(10)
    # print("start tracking")
    # system_state.value='t'
    buffered_frame=np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)

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
        if num_event.value==0:
            continue
        event_data= event_data[:num_event.value]
        rgb_image=flir_data
        event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame,accumulation_time_us=1000)
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
            # cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
            # cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
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
    parser.add_argument('--record_index', type=int, default=0,
                       help='Initial record index (default: 40)')
    parser.add_argument('--test_mode', type=str, default="Event",
                        help='Test mode (default: Model). Options: Model, Event, RGB, Show')
    parser.add_argument('--task', type=str, default="normal",
                        choices=["normal", "dark", "clutter", "dark_clutter"],
                                help='Task configuration to use (default: normal)')
    args = parser.parse_args()
    
    # Set environment variables for config selection
    config_mapping = {
        "normal": "config/config_l.yaml",
        "dark": "config/config_d.yaml",
        "clutter": "config/config_c.yaml",
        "dark_clutter": "config/config_dc.yaml"
    }
    
    # Set environment variables for subprocess config loading
    if args.task in config_mapping:
        os.environ['ACTIVE_CONFIG_FILE'] = config_mapping[args.task]
        os.environ['ACTIVE_TASK'] = args.task
        print(f"使用任务配置 {args.task}: {config_mapping[args.task]}")
    else:
        print(f"未知任务类型: {args.task}, 使用默认配置")
    
    device = torch.device("cuda")

    # Check if GUI should be enabled
    USE_GUI = os.environ.get('ENABLE_GUI', '1') == '1'
    print(f"GUI Display: {'Enabled' if USE_GUI else 'Disabled'}")

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
    event_shape=(100000,)
    event_memory=shared_memory.SharedMemory(create=True,size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data=(event_memory,event_dtype,event_shape)

    flir_dtype=np.uint8
    flir_shape=(1440,1080,3)
    flir_memory = shared_memory.SharedMemory(create=True, size=np.prod(flir_shape) * np.dtype(flir_dtype).itemsize)
    shared_flir_data=(flir_memory,flir_dtype,flir_shape)
    # stop_event already initialized above

    ROI_dtype=np.uint16
    ROI_shape=(4,)
    ROI_memory=shared_memory.SharedMemory(create=True, size=np.prod(ROI_shape) * np.dtype(ROI_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_memory.buf)
    ROI[:]=np.zeros(4)
    shared_ROI_data=(ROI_memory,ROI_dtype,ROI_shape)

    ball_pixel_dtype = np.float64
    ball_pixel_shape = (100,3)
    ball_pixel_memory = shared_memory.SharedMemory(create=True, size=np.prod(ball_pixel_shape) * np.dtype(ball_pixel_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ball_pixel = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_memory.buf)
    ball_pixel[:] = np.zeros(ball_pixel_shape)
    shared_ball_pixel_data = (ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape)

    ball_pixel_3D_dtype = np.float64
    ball_pixel_3D_shape = (1000, 4)
    ball_pixel_3D_memory = shared_memory.SharedMemory(create=True,
                                                   size=np.prod(ball_pixel_3D_shape) * np.dtype(ball_pixel_3D_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_memory.buf)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    shared_ball_pixel_3D_data = (ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape)

    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True, size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    impact_point= np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(3)
    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    keyboard_input = Value('u', 'e')
    system_state=Value('u', 'w')
    event_broadcasting = Value('u', 's')
    video_broadcasting = Value('u', 's')
    ball_observations = Value('i',0)
    raw_count = Value('i', 0)
    num_event = Value('i', 0)
    record_index = Value('i', args.record_index)
    start_ts = Value('d', 0.0)

    # Data version counters to prevent stale data processing
    event_data_version = Value('i', 0)  # Incremented each time event data is updated
    flir_data_version = Value('i', 0)   # Incremented each time FLIR data is updated

    print(f"Starting with record_index: {args.record_index}")

    # Get record directory from config
    from config.config_loader import Config
    paths = Config.get_paths()
    record_dir = paths['record_dir']
    print(f"Record directory: {record_dir}")

    # Define all available processes
    all_processes = {
        'event_process': Process(target=read_from_raw, args=(shared_event_data,event_data_version,system_state,num_event,raw_count,record_index,start_ts,stop_event,event_broadcasting,record_dir)),
        'flir_process': Process(target=read_from_video, args=(shared_flir_data,flir_data_version,system_state,raw_count,record_index,start_ts, stop_event,video_broadcasting,record_dir)),
        'model_ROI_process': Process(target=model, args=(shared_event_data,shared_flir_data,shared_ROI_data,num_event,event_data_version,flir_data_version,system_state, stop_event)),
        'RGB_ROI_process': Process(target=pure_RGB,args=(shared_event_data, shared_flir_data, shared_ROI_data, num_event,event_data_version,flir_data_version,system_state, stop_event)),
        'event_tracking_process': Process(target=event_tracking_3D,args=(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_event,event_data_version,flir_data_version,system_state, stop_event)),
        'RGB_tracking_process': Process(target=RGB_tracking_3D, args=(shared_event_data, shared_flir_data, shared_ROI_data, shared_ball_pixel_3D_data, ball_observations, num_event,event_data_version,flir_data_version,system_state, stop_event)),
        'landing_prediction_process': Process(target=landing_prediction, args=(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,record_index,system_state,stop_event)),
        'keyboard_process': Process(target=keyboard, args=(keyboard_input,system_state)),
        'robot_control_process': Process(target=robot_control, args=(shared_impact_point_data,system_state,stop_event)),
        'checker_process': Process(target=check_align,args=(shared_event_data,shared_flir_data,num_event,event_data_version,system_state,stop_event)),
        'reset_process': Process(target=reseter_for_multi_record, args=(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations, shared_impact_point_data,raw_count,num_event,record_index,start_ts,event_broadcasting,video_broadcasting,event_data_version,flir_data_version,system_state,stop_event)),
        'starter_process': Process(target=starter, args=(system_state,stop_event)),
        'gui_process': Process(target=gui_display_process,
                              args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                    shared_ball_pixel_3D_data, ball_observations, num_event,
                                    event_data_version, flir_data_version, system_state, stop_event)),
    }
    
    # Define process groups for different functionalities
    # YOU CAN MODIFY THESE GROUPS OR ADD NEW ONES
    process_groups = {
        'Model': ['event_process', 'flir_process', 'model_ROI_process', 'event_tracking_process','landing_prediction_process', 'reset_process', 'gui_process'],
        'Event': ['event_process', 'flir_process', 'event_tracking_process','landing_prediction_process','starter_process', 'reset_process', 'gui_process'],
        'RGB': ['event_process', 'flir_process', 'RGB_ROI_process', 'RGB_tracking_process', 'landing_prediction_process', 'reset_process','starter_process', 'gui_process'],
        'Show': ['event_process', 'flir_process', 'checker_process','starter_process','reset_process', 'gui_process']
    }
    # Get processes to start for current mode
    processes_to_start = process_groups.get(args.test_mode, process_groups[args.test_mode])
    print(f"Starting mode: {args.test_mode}")
    print(f"Processes to start: {processes_to_start}")

    # Start selected processes dynamically
    # started_processes is already initialized as global variable
    for process_name in processes_to_start:
        if process_name in all_processes:
            # GUI process needs environment variable check
            if process_name == 'gui_process' and not USE_GUI:
                print(f"Skipped: {process_name} (GUI disabled)")
                continue

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

