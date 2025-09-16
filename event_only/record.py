
"""
Event-Only Ball Catching System

A specialized system for real-time ball detection, tracking, and catching using event cameras.
Implements trajectory prediction and robot control for autonomous ball catching.

Features:
- Event-based ball detection and tracking
- Real-time trajectory fitting and landing prediction
- Robot motor control for ball catching
- Configurable parameters for different scenarios
"""

import os
import sys
import torch
from multiprocessing import Process, shared_memory, Value
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
from config_loader import switch_global_config

def starter(system_state, stop_event):
    """
    Initialize and start the event tracking system after a brief delay.
    
    Args:
        system_state: Shared system state variable
        stop_event: Event to signal process termination
    """
    time.sleep(3)
    print("EVENT SYSTEM: Starting event-based ball tracking...")
    system_state.value = 't'

# Global variables for signal handler
started_processes = []
stop_event = None

def signal_handler(sig, frame):
    """
    Handle system signals for graceful shutdown.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    print("\nEVENT SYSTEM: Signal received, initiating safe shutdown...")
    global stop_event, started_processes
    if stop_event:
        stop_event.set()
    shutdown_processes(started_processes)
    sys.exit(0)

def keyboard(shared_value, system_state):
    """
    Handle keyboard input for system control.
    
    Args:
        shared_value: Shared keyboard input variable
        system_state: Shared system state variable
    """
    try:
        from pynput import keyboard
    except ImportError:
        print("EVENT SYSTEM: Keyboard control not available (pynput not installed)")
        return
        
    def on_press(key):
        try:
            print(f"EVENT SYSTEM: Key {key} pressed")
            if key.char == 't':
                system_state.value = 't'
        except AttributeError:
            system_state.value = 'e'

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def reseter_for_multi_record(shared_event_data, shared_ball_pixel_3D_data, ball_observations,
                             shared_impact_point_data, num_event, record_index, system_state, stop_event):
    """
    Handle system reset between multiple recording sessions.
    
    Args:
        shared_event_data: Shared memory for event data
        shared_ball_pixel_3D_data: Shared memory for ball tracking data
        ball_observations: Ball observation counter
        shared_impact_point_data: Shared memory for impact predictions
        num_event: Number of events counter
        record_index: Current record index
        system_state: Current system state
        stop_event: Event to signal process termination
    """
    # Initialize shared memory handles
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)

    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        
        if system_state.value == 'r':
            print("EVENT SYSTEM: Initiating system reset...")
            time.sleep(2)
            system_state.value = 'w'
            print("EVENT SYSTEM: Set to waiting state")
            time.sleep(2)
            print("EVENT SYSTEM: Clearing shared memory...")
            
            # Clear all shared memory arrays
            event_array = np.ndarray(event_shape, dtype=np.dtype(event_dtype), buffer=event_shm.buf)
            event_array[:] = np.zeros(event_shape, dtype=np.dtype(event_dtype))

            ball_pixel_3D_array = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype,
                                             buffer=ball_pixel_3D_shm.buf)
            ball_pixel_3D_array[:] = 0

            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            impact_point_array[:] = 0

            # Reset counters
            ball_observations.value = 0
            num_event.value = 0
            
            print("EVENT SYSTEM: Memory cleared, restarting...")
            time.sleep(2)
            record_index.value += 1
            system_state.value = 'e'

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

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Event-Only Ball Catching System - Real-time ball detection and catching using event cameras')
    parser.add_argument('--record_index', type=int, default=0,
                       help='Initial record index for data playback (default: 0)')
    parser.add_argument('--task', type=str, default="normal",
                        choices=["normal"],
                        help='Task configuration preset (default: normal)')
    args = parser.parse_args()
    
    # Load configuration based on task parameter
    config_mapping = {
        "normal": "config.yaml",
    }
    
    # Load task-specific configuration
    try:
        switch_global_config(config_mapping[args.task])
        print(f"EVENT SYSTEM: Using task configuration '{args.task}': {config_mapping[args.task]}")
    except Exception as e:
        print(f"EVENT SYSTEM: Configuration loading failed: {e}, using default configuration")
    
    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    # Initialize stop_event for signal handler
    stop_event = multiprocessing.Event()
    started_processes.clear()  # Reset the list
    
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    
    # Shared objects - Event only
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape=(100000,)
    event_memory=shared_memory.SharedMemory(create=True,size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data=(event_memory,event_dtype,event_shape)

    # Create dummy FLIR data for compatibility with tracking module
    flir_dtype = np.uint8
    flir_shape = (1440, 1080, 3)
    flir_memory = shared_memory.SharedMemory(create=True, size=np.prod(flir_shape) * np.dtype(flir_dtype).itemsize)
    shared_flir_data = (flir_memory, flir_dtype, flir_shape)

    # Create ROI data (Region of Interest for ball tracking)
    ROI_dtype = np.uint16
    ROI_shape = (4,)
    ROI_memory = shared_memory.SharedMemory(create=True, size=np.prod(ROI_shape) * np.dtype(ROI_dtype).itemsize)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_memory.buf)
    ROI[:] = np.zeros(4)  # No ROI filtering by default
    shared_ROI_data = (ROI_memory, ROI_dtype, ROI_shape)

    ball_pixel_3D_dtype = np.float64
    ball_pixel_3D_shape = (1000, 4)
    ball_pixel_3D_memory = shared_memory.SharedMemory(create=True,
                                                   size=np.prod(ball_pixel_3D_shape) * np.dtype(ball_pixel_3D_dtype).itemsize)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_memory.buf)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    shared_ball_pixel_3D_data = (ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape)

    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True, size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize)
    impact_point= np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(3)
    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    keyboard_input = Value('u', 'e')
    system_state=Value('u', 'w')
    ball_observations = Value('i',0)
    raw_count = Value('i', 0)
    num_event = Value('i', 0)
    record_index = Value('i', args.record_index)
    start_ts = Value('d', 0.0)
    event_broadcasting = Value('u', 's')
    
    print(f"EVENT SYSTEM: Starting event-only system with record_index: {args.record_index}")
    
    # Load record directory from configuration
    from config_loader import Config
    paths = Config.get_paths()
    record_dir = paths['record_dir']
    print(f"EVENT SYSTEM: Record directory: {record_dir}")

    # Define Event-only processes
    all_processes = {
        'event_process': Process(target=read_from_event, args=(shared_event_data,system_state,num_event,stop_event)),
        'event_tracking_process': Process(target=event_tracking_3D,args=(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_event,system_state, stop_event)),
        'landing_prediction_process': Process(target=landing_prediction, args=(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,record_index,system_state,stop_event)),
        'keyboard_process': Process(target=keyboard, args=(keyboard_input,system_state)),
        'robot_control_process': Process(target=robot_control, args=(shared_impact_point_data,system_state,stop_event)),
        'reset_process': Process(target=reseter_for_multi_record, args=(shared_event_data,shared_ball_pixel_3D_data,ball_observations, shared_impact_point_data,num_event,record_index,system_state,stop_event)),
        'starter_process': Process(target=starter, args=(system_state,stop_event))
    }
    
    # Define event-only process group
    processes_to_start = ['event_process', 'event_tracking_process', 'landing_prediction_process', 'reset_process', 'starter_process']
    
    print("EVENT SYSTEM: Starting event-only mode")
    print(f"EVENT SYSTEM: Processes to start: {processes_to_start}")

    # Start selected processes
    for process_name in processes_to_start:
        if process_name in all_processes:
            process = all_processes[process_name]
            process.start()
            started_processes.append(process)
            print(f"EVENT SYSTEM: Started {process_name}")
        else:
            print(f"EVENT SYSTEM: Warning - Process '{process_name}' not found!")
    
    # Wait for processes to finish or system_state becomes 'e'
    try:
        while True:
            # Check for exit signal
            if system_state.value == 'e':
                print("EVENT SYSTEM: Exit requested, terminating processes...")
                stop_event.set()
                break
            
            # Check if all processes are still running
            if not any(p.is_alive() for p in started_processes):
                print("EVENT SYSTEM: All processes finished naturally")
                break
            
            time.sleep(0.1)  # Check every 100ms
        
        # Safe shutdown of all started processes
        shutdown_processes(started_processes)
        
    except KeyboardInterrupt:
        print("EVENT SYSTEM: Keyboard interrupt received, shutting down...")
        stop_event.set()
        shutdown_processes(started_processes)
    finally:
        print("EVENT SYSTEM: Cleaning up shared memory...")
        for shm in [event_memory, flir_memory, ROI_memory, 
                    ball_pixel_3D_memory, impact_point_memory]:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # Memory already cleaned up
                pass
        print("EVENT SYSTEM: Shutdown complete.")