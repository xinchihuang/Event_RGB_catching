import os
import sys
import torch
from multiprocessing import Process, Value
import multiprocessing
import signal
import time
import argparse
from process_lifecycle import run_workers

# Add project path for module imports
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

# Import custom modules
from data_input.read_from_event import read_from_event
from data_input.read_from_flir import read_from_flir
from tracking.ROI_generation.AMTTrack_ROI import model
from RGB_baseline.RGB_ROI import pure_RGB
from RGB_baseline.HSV_tracker import RGB_tracking_3D
from tracking.Event_tracker import event_tracking_3D_direct
from tracking.fitting.fitting_process import landing_prediction
from hardware.motor import robot_control
from gui_process import gui_display_process
# Import configuration
from shared_memory_manager import SharedMemoryManager

# Shared stop request used by the parent signal handler
stop_event = None


def signal_handler(sig, frame):
    print("\nSignal received, initiating safe shutdown...")
    # The main loop performs shutdown and cleanup exactly once.
    if stop_event:
        stop_event.set()


def keyboard(shared_value,system_state):
    from pynput import keyboard
    import threading
    import time
    
    def play_voice_notification(message):
        """Play voice notification"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)  # Speech rate
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
        while system_state.value != 'e':
            current_state = system_state.value
            if previous_state != current_state and current_state == 't':
                print(f"State change detected: {previous_state} -> {current_state}")
                # Play voice in new thread to avoid blocking
                voice_thread = threading.Thread(
                    target=play_voice_notification, 
                    args=("System started tracking. Throw the ball now!",),
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

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Event RGB Catching System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--record_index', type=int, default=0,
                       help='Initial record/output index')
    parser.add_argument('--test_mode', type=str, default="Model",
                        help='Test mode. Options: Model, Event, RGB')
    parser.add_argument('--task', type=str, default="dark_clutter",
                        choices=["normal", "dark", "clutter", "dark_clutter"],
                                help='Task configuration to use')
    args = parser.parse_args()

    # Fail before shared-memory allocation or worker startup.
    from config.config_loader import initialize_config
    try:
        active_config = initialize_config(args.task, args.test_mode)
    except (OSError, ValueError) as exc:
        parser.error(f"Configuration error: {exc}")
    from config.config_loader import Config
    try:
        startup_paths = Config.get_paths()
    except (OSError, ValueError) as exc:
        parser.error(f"Data path error: {exc}")
    print(f"Evaluation output: {startup_paths['evaluation_root']}", flush=True)

    
    # Set environment variables for GUI process to access test_mode and task
    os.environ['TEST_MODE'] = args.test_mode
    print(f"设置测试模式: {args.test_mode}, 任务: {args.task}")
    from config.config_loader import get_current_config_name
    print(f"Active config overlay: {get_current_config_name()}")
    
    device = torch.device("cuda")

    # Check if GUI should be enabled
    USE_GUI = os.environ.get('ENABLE_GUI', '1') == '1'
    print(f"GUI Display: {'Enabled' if USE_GUI else 'Disabled'}")

    multiprocessing.set_start_method('spawn')

    # Initialize stop_event for signal handler
    stop_event = multiprocessing.Event()
    
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
                                                    shared_impact_point_data, record_index, system_state, stop_event)),
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
        'Event': ['event_process','event_tracking_process','landing_prediction_process','keyboard_process','gui_process'],
        'RGB': ['event_process','flir_process','RGB_ROI_process', 'RGB_tracking_process', 'landing_prediction_process', 'keyboard_process','gui_process'],
        'Model':['event_process','flir_process','model_ROI_process','event_tracking_process','landing_prediction_process','keyboard_process','gui_process'],
    }
    # Get processes to start for current mode
    processes_to_start = process_groups.get(args.test_mode, process_groups[args.test_mode])
    print(f"Starting mode: {args.test_mode}")
    print(f"Processes to start: {processes_to_start}")

    exit_code = 1
    try:
        selected = [(name, all_processes[name]) for name in processes_to_start
                    if name != 'gui_process' or USE_GUI]
        exit_code = run_workers(selected, stop_event, system_state)
    finally:
        # Workers have stopped before their shared memory is released.
        shm_manager.cleanup_all()
    sys.exit(exit_code)
