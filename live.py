#!/usr/bin/env python3
"""
Live Event-Only Ball Catching System

A simplified real-time ball detection, tracking, and catching system using event cameras.
Based on the main record.py structure but optimized for live event camera feeds.
Implements trajectory prediction and robot control for autonomous ball catching.

Features:
- Live event camera data processing
- Real-time ball detection and tracking
- Physics-based trajectory prediction
- Robot motor control for ball catching
- Event-only testing mode
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

# Add project path for module imports
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import event-only modules
from data_input.read_from_event import read_from_event
from tracking_2D.Event_tracker import event_tracking_3D
from fitting.fitting_process import landing_prediction
from hardware.motor import robot_control

# Import configuration
from config.config_loader import Config

def starter(system_state, stop_event):
    """
    Initialize and start the event-only tracking system after a brief delay.
    
    Args:
        system_state: Shared system state variable
        stop_event: Event to signal process termination
    """
    time.sleep(3)
    print("LIVE SYSTEM: Starting event-only ball tracking...")
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
    print("\nLIVE SYSTEM: Signal received, initiating safe shutdown...")
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
        print("LIVE SYSTEM: No processes to shutdown.")
        return
    
    print(f"LIVE SYSTEM: Shutting down {len(processes)} processes...")
    
    # Phase 1: Graceful shutdown
    print("LIVE SYSTEM: Phase 1 - Waiting for graceful shutdown...")
    start_time = time.time()
    
    while processes and (time.time() - start_time) < timeout:
        alive_processes = [p for p in processes if p.is_alive()]
        if not alive_processes:
            print("LIVE SYSTEM: All processes exited gracefully.")
            return
        
        print(f"LIVE SYSTEM: Waiting for {len(alive_processes)} processes to exit...")
        time.sleep(0.5)
    
    # Phase 2: Force termination
    alive_processes = [p for p in processes if p.is_alive()]
    if alive_processes:
        print(f"LIVE SYSTEM: Phase 2 - Force terminating {len(alive_processes)} remaining processes...")
        
        for process in alive_processes:
            try:
                print(f"LIVE SYSTEM: Terminating process {process.pid}...")
                process.terminate()
            except Exception as e:
                print(f"LIVE SYSTEM: Error terminating process {process.pid}: {e}")
        
        time.sleep(1)
        
        # Phase 3: Force kill if necessary
        still_alive = [p for p in alive_processes if p.is_alive()]
        if still_alive:
            print(f"LIVE SYSTEM: Phase 3 - Force killing {len(still_alive)} stubborn processes...")
            
            for process in still_alive:
                try:
                    print(f"LIVE SYSTEM: Killing process {process.pid}...")
                    process.kill()
                except Exception as e:
                    print(f"LIVE SYSTEM: Error killing process {process.pid}: {e}")
        
        # Final status check
        time.sleep(0.5)
        final_alive = [p for p in processes if p.is_alive()]
        if final_alive:
            print(f"LIVE SYSTEM: Warning - {len(final_alive)} processes still running after shutdown!")
        else:
            print("LIVE SYSTEM: All processes successfully terminated.")
    
    print("LIVE SYSTEM: Process shutdown complete.")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Live Event-Only Ball Catching System - Real-time ball detection using Event cameras')
    parser.add_argument('--record_index', type=int, default=0,
                       help='Initial record index for tracking (default: 0)')
    parser.add_argument('--task', type=str, default="normal",
                        choices=["normal", "dark", "cluster", "dark_cluster"],
                        help='Task configuration to use (default: normal)')
    args = parser.parse_args()
    
    # Load configuration based on task parameter
    config_mapping = {
        "model": {
            "normal": "config/config_l_model.yaml",
            "dark": "config/config_d_model.yaml",
            "cluster": "config/config_c_model.yaml",
            "dark_cluster": "config/config_dc_model.yaml"
        },
        "rgb": {
            "normal": "config/config_l_rgb.yaml",
            "dark": "config/config_d_rgb.yaml",
            "cluster": "config/config_c_rgb.yaml",
            "dark_cluster": "config/config_dc_rgb.yaml"
        },
        "event": {
            "normal": "config/config_l_event.yaml",
            "dark": "config/config_d_event.yaml",
            "cluster": "config/config_c_event.yaml",
            "dark_cluster": "config/config_dc_event.yaml"
        }
    }
    legacy_mapping = {
        "normal": "config/config.yaml",
        "dark": "config/config_d.yaml",
        "cluster": "config/config_c.yaml",
        "dark_cluster": "config/config_dc.yaml"
    }
    mode_key = os.environ.get("TEST_MODE", "").strip().lower()
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
        print(f"LIVE SYSTEM: Using task configuration '{args.task}': {config_file}")
    else:
        print(f"LIVE SYSTEM: Unknown task type: {args.task}, using default configuration")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LIVE SYSTEM: Using device: {device}")

    multiprocessing.set_start_method('spawn')

    # Initialize stop_event for signal handler
    stop_event = multiprocessing.Event()
    started_processes.clear()  # Reset the list
    
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    
    # Shared objects - Event data
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape = (100000,)
    event_memory = shared_memory.SharedMemory(create=True, size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data = (event_memory, event_dtype, event_shape)

    # Create dummy FLIR data for compatibility with tracking module
    flir_dtype = np.uint8
    flir_shape = (1440, 1080, 3)
    flir_memory = shared_memory.SharedMemory(create=True, size=np.prod(flir_shape) * np.dtype(flir_dtype).itemsize)
    shared_flir_data = (flir_memory, flir_dtype, flir_shape)

    # ROI data (Region of Interest for ball tracking)
    ROI_dtype = np.uint16
    ROI_shape = (4,)
    ROI_memory = shared_memory.SharedMemory(create=True, size=np.prod(ROI_shape) * np.dtype(ROI_dtype).itemsize)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_memory.buf)
    ROI[:] = np.zeros(4)  # No ROI filtering by default
    shared_ROI_data = (ROI_memory, ROI_dtype, ROI_shape)

    # Ball tracking data (3D tracking)
    ball_pixel_3D_dtype = np.float64
    ball_pixel_3D_shape = (1000, 4)
    ball_pixel_3D_memory = shared_memory.SharedMemory(create=True,
                                                   size=np.prod(ball_pixel_3D_shape) * np.dtype(ball_pixel_3D_dtype).itemsize)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_memory.buf)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    shared_ball_pixel_3D_data = (ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape)

    # Impact point prediction data
    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True, size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize)
    impact_point = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(3)
    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    # System control variables
    system_state = Value('u', 'w')
    ball_observations = Value('i', 0)
    num_event = Value('i', 0)
    record_index = Value('i', args.record_index)
    
    print(f"LIVE SYSTEM: Starting live event-only system with record_index: {args.record_index}")

    # Get record directory from config (not used for live feeds but kept for compatibility)
    paths = Config.get_paths()
    record_dir = paths['record_dir']
    print(f"LIVE SYSTEM: Configuration record directory: {record_dir}")

    # Define the 4 essential event-only processes + starter
    all_processes = {
        'event_process': Process(target=read_from_event, 
                               args=(shared_event_data, system_state, num_event, stop_event)),
        'event_tracking_process': Process(target=event_tracking_3D,
                                        args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                              shared_ball_pixel_3D_data, ball_observations, num_event, 
                                              system_state, stop_event)),
        'landing_prediction_process': Process(target=landing_prediction, 
                                            args=(shared_ball_pixel_3D_data, ball_observations,
                                                  shared_impact_point_data, record_index, system_state, stop_event, "Live", args.task)),
        'robot_control_process': Process(target=robot_control, 
                                       args=(shared_impact_point_data, system_state, stop_event)),
        'starter_process': Process(target=starter, 
                                 args=(system_state, stop_event)),
    }
    
    # Event-only testing process group
    processes_to_start = ['event_process', 'event_tracking_process', 'landing_prediction_process', 'robot_control_process', 'starter_process']
    
    print("LIVE SYSTEM: Starting event-only testing mode")
    print(f"LIVE SYSTEM: Processes to start: {processes_to_start}")

    # Start selected processes
    for process_name in processes_to_start:
        if process_name in all_processes:
            process = all_processes[process_name]
            process.start()
            started_processes.append(process)
            print(f"LIVE SYSTEM: Started {process_name}")
        else:
            print(f"LIVE SYSTEM: Warning - Process '{process_name}' not found!")
    
    # Wait for processes to finish or system_state becomes 'e'
    try:
        while True:
            # Check for exit signal
            if system_state.value == 'e':
                print("LIVE SYSTEM: Exit requested, terminating processes...")
                stop_event.set()
                break
            
            # Check if all processes are still running
            if not any(p.is_alive() for p in started_processes):
                print("LIVE SYSTEM: All processes finished naturally")
                break
            
            time.sleep(0.1)  # Check every 100ms
        
        # Safe shutdown of all started processes
        shutdown_processes(started_processes)
        
    except KeyboardInterrupt:
        print("LIVE SYSTEM: Keyboard interrupt received, shutting down...")
        stop_event.set()
        shutdown_processes(started_processes)
    finally:
        print("LIVE SYSTEM: Cleaning up shared memory...")
        for shm in [event_memory, flir_memory, ROI_memory, 
                    ball_pixel_3D_memory, impact_point_memory]:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # Memory already cleaned up
                pass
        print("LIVE SYSTEM: Shutdown complete.")
