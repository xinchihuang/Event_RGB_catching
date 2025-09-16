"""
Event-based ball tracking module for real-time ball detection and tracking.
Implements DBSCAN clustering for ball detection from event camera data.
"""

import cv2
import os
import sys
from multiprocessing import Process, Array, Queue, Manager, shared_memory, Value
import numpy as np
import time
import pandas as pd

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def ROI_filter(events, ROI_data):
    """
    Filter events to only include those within a specified Region of Interest (ROI).

    Args:
        events (np.ndarray): Structured array of events with x, y coordinates
        ROI_data (tuple): ROI parameters (x_r, y_r, w_r, h_r) defining the region

    Returns:
        np.ndarray: Filtered events within the ROI bounds
    """
    roi_params = Config.get_roi()
    roi_range = roi_params['roi_range']
    x_r, y_r, w_r, h_r = ROI_data
    
    # If ROI is not defined, return all events
    if x_r == 0 and y_r == 0 and w_r == 0 and h_r == 0:
        return events
        
    # Expand ROI bounds for better tracking
    x_r = max(x_r - roi_range, 0)
    y_r = max(y_r - 2*roi_range, 0)
    w_r = w_r + 2 * roi_range
    h_r = h_r + 4 * roi_range
    
    # Apply spatial filter
    events_cp = events.copy()
    mask = (
        (events_cp['x'] >= x_r) & (events_cp['x'] < x_r + w_r) &
        (events_cp['y'] >= y_r) & (events_cp['y'] < y_r + h_r)
    )
    
    return events_cp[mask]
def event_tracking_3D(shared_event_data, shared_flir_data, shared_ROI_data, shared_ball_pixel_3D_data, ball_observations, num_events, system_state, stop_event):
    """
    Main event-based ball tracking function using 3D clustering.
    
    Args:
        shared_event_data: Shared memory containing event data
        shared_flir_data: Shared memory for FLIR image data (compatibility)
        shared_ROI_data: Shared memory containing ROI parameters
        shared_ball_pixel_3D_data: Shared memory for ball position data
        ball_observations: Counter for ball observations
        num_events: Number of events in current frame
        system_state: Current system state
        stop_event: Event to signal process termination
    """
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']
    
    # Get image processing parameters from config
    processing = Config.get_image_processing()
    
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=processing['fps'],
                                                       palette=ColorPalette.Dark)


    ball_pixel_3D_memory,ball_pixel_3D_dtype,ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    # Initialize DBSCAN clustering for ball detection
    tracking = Config.get_tracking()
    dbscan = DBSCAN(eps=tracking['dbscan_eps'], min_samples=tracking['dbscan_min_samples'])
    count = 0

    # Wait for system to start tracking
    while system_state.value == 'w':
        continue
    print("EVENT TRACKING: Starting event-based ball tracking")
    
    # Main tracking loop
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        if system_state.value == 't':
            start_t = time.time()
            try:
                # Read event data from shared memory
                raw_event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
                flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
                
                if num_events.value == 0:
                    continue
                    
                # Apply ROI filtering
                raw_event_data = raw_event_data[:num_events.value]
                ROI_data = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                x_r, y_r, w_r, h_r = ROI_data
                event_data = ROI_filter(raw_event_data, ROI_data)
                
                # Extract timestamp
                t = event_data['t'][-1] / 1000000
                
                # Generate event frame for visualization
                event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
                event_frame = np.ascontiguousarray(event_frame)
                event_frame_gen.generate_frame(event_data, event_frame)
                
                # Convert to grayscale and apply threshold
                image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(image, processing['binary_threshold'], processing['binary_max_value'], cv2.THRESH_BINARY)
                
                # Find event points for clustering
                points = np.column_stack(np.where(binary > 0))
                
                # Apply DBSCAN clustering to detect ball
                dbscan.fit(points)
                labels = dbscan.labels_
                event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                cluster_points = points[labels != -1]
                
                # Process detected ball cluster
                if len(cluster_points) > 0:
                    # Calculate bounding box
                    y = min(cluster_points[:, 0])
                    x = min(cluster_points[:, 1])
                    w = max(cluster_points[:, 0]) - y
                    h = max(cluster_points[:, 1]) - x
                    
                    # Draw detection rectangle
                    cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
                    
                    # Store ball observation
                    ball_num = ball_observations.value
                    ball_pixel_3D[ball_num][0] = float(x + h / 2)  # Center X
                    ball_pixel_3D[ball_num][1] = float(y + h / 2)  # Center Y
                    ball_pixel_3D[ball_num][2] = float(h)          # Size
                    ball_pixel_3D[ball_num][3] = t                 # Timestamp
                    
                    # Print tracking information
                    center_x = x + h / 2
                    center_y = y + h / 2
                    print(f"EVENT TRACKING: Ball detected at pixel ({center_x:.1f}, {center_y:.1f}), size={h:.1f}, time={t:.3f}s")
                    print(f"EVENT TRACKING: Ball observation #{ball_num+1}, total observations: {ball_num+1}")

                    # Update observation counter
                    if ball_num < tracking['max_ball_buffer']:
                        ball_num += 1
                    ball_observations.value = ball_num

                # Display combined view
                rgb_image = flir_data
                combined = np.hstack((rgb_image, event_image))
                scale = processing['display_scale']

                # Resize for display
                width = int(combined.shape[1] * scale)
                height = int(combined.shape[0] * scale)
                new_size = (width, height)

                resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('Event Ball Tracking', resized_img)
                cv2.waitKey(1)

                count += 1
                
            except Exception as e:
                print(f"EVENT TRACKING: Error in tracking loop: {e}")
                continue