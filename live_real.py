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
from data_input.read_from_event import read_from_event
from data_input.read_from_flir import read_from_flir
from ROI_generation.AMTTrack_ROI import model
from ROI_generation.RGB_ROI import pure_RGB
from tracking_2D.HSV_tracker import RGB_tracking_3D
from tracking_2D.Event_tracker import event_tracking_3D_direct
from fitting.fitting_process import landing_prediction
from hardware.motor import robot_control
from gui_process import gui_display_process
# Import configuration
from config.config_loader import Config
from shared_memory_manager import SharedMemoryManager

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
    import queue
    import threading
    import time

    voice_messages = queue.Queue()

    def voice_worker():
        """Keep one TTS engine alive so its native callbacks remain valid."""
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.setProperty('volume', 0.9)
        except Exception as exc:
            print(f"Voice engine unavailable: {exc}")

        try:
            while True:
                message = voice_messages.get()
                if message is None:
                    break
                if engine is None:
                    print(f"Voice notification: {message}")
                    continue
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"Voice playback failed: {exc}; message: {message}")
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    voice_thread = threading.Thread(target=voice_worker, daemon=True)
    voice_thread.start()
    
    def state_monitor():
        """Monitor state changes and trigger voice notifications"""
        previous_state = system_state.value
        while system_state.value != 'e':
            current_state = system_state.value
            if previous_state != current_state and current_state == 't':
                print(f"State change detected: {previous_state} -> {current_state}")
                voice_messages.put("System started tracking. Throw the ball now!")
            previous_state = current_state
            time.sleep(0.1)
    
    # Start state monitoring thread
    monitor_thread = threading.Thread(target=state_monitor, daemon=True)
    monitor_thread.start()
    
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
            # shared_value.value = key.char if hasattr(key, 'char') else str(key)
            if key.char=='t':
                print("=== STATE CHANGE: -> 't' (Manual tracking start) ===")
                system_state.value='t'
            elif key.char=='w':
                print("=== STATE CHANGE: -> 'w' (Manual wait) ===")
                system_state.value='w'
            elif key.char == 'e':
                print("=== STATE CHANGE: -> 'e' (Manual exit) ===")
                system_state.value = 'e'
        except AttributeError:
            print("=== STATE CHANGE: -> 'e' (Exit via special key) ===")
            system_state.value='e'

    with keyboard.Listener(on_press=on_press) as listener:
        while listener.is_alive() and system_state.value != 'e':
            time.sleep(0.1)
        listener.stop()

    voice_messages.put(None)
    voice_thread.join(timeout=3)
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
                        help='Test mode (default: Model). Options: Model, Event, RGB')
    parser.add_argument('--task', type=str, default="normal",
                        choices=["normal", "dark", "clutter", "dark_cluster"],
                                help='Task configuration to use (default: normal)')
    args = parser.parse_args()
    
    # Load configuration based on task and test_mode
    config_mapping = {
        "model": {
            "normal": "config/config_l_model.yaml",
            "dark": "config/config_d_model.yaml",
            "clutter": "config/config_c_model.yaml",
            "dark_cluster": "config/config_dc_model.yaml"
        },
        "rgb": {
            "normal": "config/config_l_rgb.yaml",
            "dark": "config/config_d_rgb.yaml",
            "clutter": "config/config_c_rgb.yaml",
            "dark_cluster": "config/config_dc_rgb.yaml"
        },
        "event": {
            "normal": "config/config_l_event.yaml",
            "dark": "config/config_d_event.yaml",
            "clutter": "config/config_c_event.yaml",
            "dark_cluster": "config/config_dc_event.yaml"
        }
    }
    legacy_mapping = {
        "normal": "config/config_l.yaml",
        "dark": "config/config_d.yaml",
        "clutter": "config/config_c.yaml",
        "dark_cluster": "config/config_dc.yaml"
    }
    mode_key = (args.test_mode or "").strip().lower()
    task_key = (args.task or "").lower()
    config_file = None
    if mode_key in config_mapping:
        config_file = config_mapping[mode_key].get(task_key)
    if config_file is None:
        config_file = legacy_mapping.get(task_key)

    # Set environment variables for subprocess config loading
    if config_file:
        os.environ['ACTIVE_CONFIG_FILE'] = config_file
        os.environ['ACTIVE_TASK'] = args.task
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
    started_processes.clear()  # Reset the list
    
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

    # Create shared memory manager and initialize all shared memory
    shm_manager = SharedMemoryManager()
    shm_manager.create_all()

    # Get shared data tuples for passing to subprocesses
    shared_event_data = shm_manager.get_shared_data_tuple('event_data')
    shared_flir_data = shm_manager.get_shared_data_tuple('flir_data')
    shared_ROI_data = shm_manager.get_shared_data_tuple('ROI_data')
    shared_ball_pixel_data = shm_manager.get_shared_data_tuple('ball_pixel')
    shared_ball_pixel_3D_data = shm_manager.get_shared_data_tuple('ball_pixel_3D')
    shared_impact_point_data = shm_manager.get_shared_data_tuple('impact_point')

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
    ball_pixel_3D_version = Value('i', 0)  # Incremented each time ball_pixel_3D is updated
    
    print(f"Starting with record_index: {args.record_index}")

    # Get record directory from config
    from config.config_loader import Config
    paths = Config.get_paths()
    record_dir = paths['record_dir']
    print(f"Record directory: {record_dir}")

    # Define all available processes
    all_processes = {
        'event_process': Process(target=read_from_event,
                                 args=(shared_event_data, system_state, num_event, event_data_version, stop_event)),
        'flir_process': Process(target=read_from_flir,
                                args=(shared_flir_data, flir_data_version, system_state, stop_event)),
        'model_ROI_process': Process(target=model,
                                     args=(shared_event_data, shared_flir_data, shared_ROI_data, num_event,
                                           event_data_version, flir_data_version, system_state, stop_event)),
        'RGB_ROI_process': Process(target=pure_RGB,
                                   args=(shared_event_data, shared_flir_data, shared_ROI_data, num_event,
                                         event_data_version, flir_data_version, system_state, stop_event)),
        'event_tracking_process': Process(target=event_tracking_3D_direct,
                                          args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                                shared_ball_pixel_3D_data, ball_observations, num_event,
                                                event_data_version, flir_data_version, system_state, stop_event,
                                                ball_pixel_3D_version)),
        'RGB_tracking_process': Process(target=RGB_tracking_3D,
                                        args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                              shared_ball_pixel_3D_data, ball_observations, num_event,
                                              event_data_version, flir_data_version, system_state, stop_event,
                                              ball_pixel_3D_version)),
        'landing_prediction_process': Process(target=landing_prediction,
                                              args=(shared_ball_pixel_3D_data, ball_observations,
                                                    shared_impact_point_data, record_index, system_state,
                                                    stop_event, args.test_mode, args.task, True)),
        'keyboard_process': Process(target=keyboard, args=(keyboard_input, system_state)),
        'robot_control_process': Process(target=robot_control,
                                         args=(shared_impact_point_data, system_state, stop_event)),
        'gui_process': Process(target=gui_display_process,
                              args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                    shared_ball_pixel_3D_data, ball_observations, num_event,
                                    event_data_version, flir_data_version, system_state, stop_event),
                              kwargs={'ball_pixel_3D_version': ball_pixel_3D_version}),
    }
    
    # Define process groups for different functionalities
    # YOU CAN MODIFY THESE GROUPS OR ADD NEW ONES
    process_groups = {
        'Event': ['event_process','event_tracking_process','landing_prediction_process','keyboard_process','robot_control_process','gui_process'],
        'RGB': ['event_process','flir_process','RGB_ROI_process', 'RGB_tracking_process', 'landing_prediction_process', 'keyboard_process','robot_control_process','gui_process'],
        'Model':['event_process','flir_process','model_ROI_process','event_tracking_process','landing_prediction_process','keyboard_process','robot_control_process','gui_process'],
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
            process.name = process_name
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

            # A failed worker must stop the whole real-hardware pipeline.
            # Otherwise the remaining GUI/keyboard workers keep this parent
            # alive and make a startup failure look like a hang.
            failed_processes = [
                p for p in started_processes
                if p.exitcode is not None and p.exitcode != 0
            ]
            if failed_processes:
                failures = ", ".join(
                    f"{p.name} (exit code {p.exitcode})"
                    for p in failed_processes
                )
                print(f"Critical subprocess failure: {failures}")
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
        # Clean up all shared memory using the manager
        shm_manager.cleanup_all()
