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

# Add local path for module imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import event-only modules
from read_from_event import read_from_event
from Event_tracker import event_tracking_3D
from fitting_process import landing_prediction, robot_control

# Import configuration
from config_loader import Config

def starter(system_state,stop_event):
    time.sleep(3)
    print("EVENT SYSTEM: Starting event-based ball tracking...")
    system_state.value = 't'

# Global variables for signal handler
started_processes = []
stop_event = None

def signal_handler(sig, frame):
    print("\nEVENT SYSTEM: Signal received, initiating safe shutdown...")
    global stop_event, started_processes
    if stop_event:
        stop_event.set()
    shutdown_processes(started_processes)
    sys.exit(0)

def shutdown_processes(processes, timeout=5):
    """
    Safely shutdown all system processes with graceful fallback to force termination.
    
    Args:
        processes (list): List of processes to shutdown
        timeout (int): Timeout for graceful exit in seconds
    """
    if not processes:
        print("EVENT SYSTEM: No processes to shutdown.")
        return
    
    print(f"EVENT SYSTEM: Shutting down {len(processes)} processes...")
    
    # Phase 1: Graceful shutdown
    print("EVENT SYSTEM: Phase 1 - Waiting for graceful shutdown...")
    start_time = time.time()
    
    while processes and (time.time() - start_time) < timeout:
        alive_processes = [p for p in processes if p.is_alive()]
        if not alive_processes:
            print("EVENT SYSTEM: All processes exited gracefully.")
            return
        
        print(f"EVENT SYSTEM: Waiting for {len(alive_processes)} processes to exit...")
        time.sleep(0.5)
    
    # Phase 2: Force termination
    alive_processes = [p for p in processes if p.is_alive()]
    if alive_processes:
        print(f"EVENT SYSTEM: Phase 2 - Force terminating {len(alive_processes)} remaining processes...")
        
        for process in alive_processes:
            try:
                print(f"EVENT SYSTEM: Terminating process {process.pid}...")
                process.terminate()
            except Exception as e:
                print(f"EVENT SYSTEM: Error terminating process {process.pid}: {e}")
        
        time.sleep(1)
        
        # Phase 3: Force kill if necessary
        still_alive = [p for p in alive_processes if p.is_alive()]
        if still_alive:
            print(f"EVENT SYSTEM: Phase 3 - Force killing {len(still_alive)} stubborn processes...")
            
            for process in still_alive:
                try:
                    print(f"EVENT SYSTEM: Killing process {process.pid}...")
                    process.kill()
                except Exception as e:
                    print(f"EVENT SYSTEM: Error killing process {process.pid}: {e}")
        
        # Final status check
        time.sleep(0.5)
        final_alive = [p for p in processes if p.is_alive()]
        if final_alive:
            print(f"EVENT SYSTEM: Warning - {len(final_alive)} processes still running after shutdown!")
        else:
            print("EVENT SYSTEM: All processes successfully terminated.")
    
    print("EVENT SYSTEM: Process shutdown complete.")
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

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Event RGB Catching System')
    parser.add_argument('--record_index', type=int, default=0,
                       help='Initial record index (default: 0)')
    parser.add_argument('--test_mode', type=str, default="Event",
                        help='Test mode. Options:Event')
    parser.add_argument('--task', type=str, default="normal",
                        choices=["normal"],
                                help='Task configuration to use (default: normal)')
    args = parser.parse_args()
    
    # Load configuration based on task parameter
    config_mapping = {
        "normal": "config.yaml",
    }
    
    # Set environment variables for subprocess config loading
    if args.task in config_mapping:
        os.environ['ACTIVE_CONFIG_FILE'] = config_mapping[args.task]
        os.environ['ACTIVE_TASK'] = args.task
        print(f"Using task configuration '{args.task}': {config_mapping[args.task]}")
    else:
        print(f"Unknown task type: {args.task}, using default configuration")
    
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
    event_shape=(100000,)
    event_memory=shared_memory.SharedMemory(create=True,size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data=(event_memory,event_dtype,event_shape)

    # Note: FLIR/RGB camera not available in event_only mode
    # Creating minimal shared memory for compatibility
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
    
    print(f"Starting with record_index: {args.record_index}")

    # Get record directory from config
    try:
        from config_loader import Config
        paths = Config.get_paths()
        record_dir = paths['record_dir']
        print(f"EVENT SYSTEM: Record directory: {record_dir}")
    except Exception as e:
        print(f"EVENT SYSTEM: Warning - Could not load record directory config: {e}")
        record_dir = "/tmp/event_records"

    # Define all available processes (only event_only compatible functions)
    all_processes = {
        'event_process': Process(target=read_from_event,
                                 args=(shared_event_data, system_state,num_event,stop_event)),
        'event_tracking_process': Process(target=event_tracking_3D,
                                          args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                                shared_ball_pixel_3D_data, ball_observations, num_event, system_state,
                                                stop_event)),
        'landing_prediction_process': Process(target=landing_prediction,
                                              args=(shared_ball_pixel_3D_data, ball_observations,
                                                    shared_impact_point_data, record_index, system_state, stop_event)),
        'keyboard_process': Process(target=keyboard, args=(keyboard_input, system_state)),
        'starter_process': Process(target=starter, args=(system_state, stop_event))
    }
    
    # Define process groups for different functionalities
    # Only event-based processes are available in event_only directory
    process_groups = {
        'Event': ['event_process','event_tracking_process','landing_prediction_process','keyboard_process'],
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
        for shm in [event_memory, flir_memory, ROI_memory,
                    ball_pixel_3D_memory, impact_point_memory]:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # Already unlinked
                pass

