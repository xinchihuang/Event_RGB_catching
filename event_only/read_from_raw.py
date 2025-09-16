"""
Event data reader module for processing .raw event files.
Handles event data alignment, coordinate transformation, and real-time playback.
"""

import cv2
import os
from multiprocessing import Process, Array, Queue, Manager, shared_memory, Value
import numpy as np
import time
import pandas as pd

# Import configuration
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def align_to_rgb(event_data, rgb_width=None, rgb_height=None, event_width=None, event_height=None):
    """
    Align event camera coordinates to RGB camera coordinate system.
    
    Args:
        event_data: Event data array with x, y coordinates
        rgb_width, rgb_height: RGB camera dimensions
        event_width, event_height: Event camera dimensions
        
    Returns:
        tuple: Aligned (x, y) coordinates as integers
    """
    # Load dimensions from configuration if not provided
    if rgb_width is None or rgb_height is None or event_width is None or event_height is None:
        dims = Config.get_image_dimensions()
        rgb_width = dims['rgb_width'] if rgb_width is None else rgb_width
        rgb_height = dims['rgb_height'] if rgb_height is None else rgb_height
        event_width = dims['event_width'] if event_width is None else event_width
        event_height = dims['event_height'] if event_height is None else event_height
    
    # Load alignment parameters from configuration
    alignment = Config.get_alignment()
    event_pixel = alignment['event_pixel_size']
    rgb_pixel = alignment['rgb_pixel_size']
    offset_x = alignment['offset_x']
    offset_y = alignment['offset_y']
    
    # Calculate coordinate transformation
    x_e, y_e = event_data['x'].copy(), event_data['y'].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2

    # Apply scaling and offset transformation
    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + offset_x
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + offset_y
    y = np.clip(y, 0, rgb_height)
    
    return x.astype(int), y.astype(int)
def read_from_raw(shared_event_data, system_state, num_event,
                  raw_count, record_index, start_ts, stop_event, event_broadcasting,
                  record_dir='', event_frame_offset=0, playback_speed=1.0, current_event_frame=None, is_paused=None):
    """
    Read and playback event data from .raw files.
    
    Processing pipeline:
    1. Pre-load entire .raw file into event_buffer (chunked by time intervals)
    2. Playback phase: Synchronized event streaming with configurable timing
    3. Coordinate transformation and alignment to RGB camera system
    
    Args:
        shared_event_data: Shared memory for event data
        system_state: Current system state
        num_event: Number of events in current frame
        raw_count: Count of raw data chunks
        record_index: Current record index
        start_ts: Start timestamp for synchronization
        stop_event: Event to signal process termination
        event_broadcasting: Broadcasting state flag
        record_dir: Directory containing raw files
        event_frame_offset: Frame offset for playback control
        playback_speed: Speed multiplier for playback
        current_event_frame: Current frame index (for debugging)
        is_paused: Pause state flag
    """

    import os, time, numpy as np
    from multiprocessing import shared_memory
    from metavision_core.event_io import EventsIterator

    # Initialize shared memory handles
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    # Load data reading parameters from configuration
    data_reading = Config.get_data_reading()
    period = data_reading['send_interval'] / playback_speed  # Apply playback speed adjustment
    event_count = 0
    initialized = False

    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # Pre-load .raw file data
        if system_state.value == 'w' and not initialized:
            raw_path = os.path.join(record_dir, f"event/{record_index.value}.raw")
            event_buffer = []
            
            # Initialize event iterator
            iterator = EventsIterator(input_path=raw_path, mode="delta_t",
                                      delta_t=data_reading['delta_t'], 
                                      n_events=data_reading['n_events'])
            
            # Process and transform event data
            for ev in iterator:
                # Apply coordinate transformation
                ev['x'], ev['y'] = ev['y'].copy(), -ev['x'].copy()
                alignment = Config.get_alignment()
                ev['y'] += alignment['y_coordinate_offset']
                ev['x'], ev['y'] = align_to_rgb(ev)
                event_buffer.append(ev)
                
            raw_count.value = len(event_buffer)  # Update count for synchronization
            print(f"EVENT READER: Loaded {raw_count.value} event chunks")

            event_count = 0
            initialized = True
            print("EVENT READER: Data pre-loading completed")
            
        # Event playback phase
        if system_state.value == 't' and initialized:
            # Initialize timing on first playback entry
            if start_ts.value == 0.0:
                start_ts.value = time.perf_counter()

            # Apply event frame offset for synchronization
            adjusted_event_count = event_count + event_frame_offset
            adjusted_event_count = max(0, min(adjusted_event_count, len(event_buffer) - 1))
            
            loop_idx = event_count
            target_time = start_ts.value + loop_idx * period

            # Check for end of data
            if loop_idx >= len(event_buffer):
                event_broadcasting.value = 'f'
                initialized = False
                print("EVENT READER: End of event data reached")
                continue

            # Process current event chunk
            ev = event_buffer[adjusted_event_count]
            if ev.shape[0] == 0:
                # Skip empty event chunks
                event_count += 1
                continue
                
            # Write events to shared memory
            n = ev.shape[0]
            num_event.value = n
            shm[:n] = ev[:min(n, 100000)]  # Limit to shared memory buffer size
            
            # Update frame index for debugging
            if current_event_frame is not None:
                current_event_frame.value = adjusted_event_count

            # Maintain precise timing
            sleep_time = target_time + period - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Handle pause state
            while is_paused and is_paused.value == 1 and not stop_event.is_set():
                time.sleep(0.1)

            event_count += 1