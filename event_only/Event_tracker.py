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

    # Timing statistics
    from datetime import datetime
    process_start_time = time.time()
    frame_times = []

    # Main tracking loop
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        if system_state.value == 't':
            # === Timing Section Start ===
            frame_start_absolute = datetime.now()
            frame_start_relative = time.time()
            frame_start_since_process = frame_start_relative - process_start_time

            try:
                # Read event data from shared memory
                data_read_start = time.time()
                raw_event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
                flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()
                data_read_time = time.time() - data_read_start

                if num_events.value == 0:
                    continue

                # Apply ROI filtering
                roi_filter_start = time.time()
                raw_event_data = raw_event_data[:num_events.value]
                ROI_data = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                x_r, y_r, w_r, h_r = ROI_data
                event_data = ROI_filter(raw_event_data, ROI_data)
                roi_filter_time = time.time() - roi_filter_start

                # Extract timestamp
                t = event_data['t'][-1] / 1000000

                # Generate event frame for visualization
                frame_gen_start = time.time()
                event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
                event_frame = np.ascontiguousarray(event_frame)
                event_frame_gen.generate_frame(event_data, event_frame)
                frame_gen_time = time.time() - frame_gen_start

                # Convert to grayscale and apply threshold
                preprocess_start = time.time()
                image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(image, processing['binary_threshold'], processing['binary_max_value'], cv2.THRESH_BINARY)
                points = np.column_stack(np.where(binary > 0))
                preprocess_time = time.time() - preprocess_start

                # Apply DBSCAN clustering to detect ball
                dbscan_start = time.time()
                dbscan.fit(points)
                labels = dbscan.labels_
                dbscan_time = time.time() - dbscan_start
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
                display_start = time.time()
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
                display_time = time.time() - display_start

                # === Timing Section End ===
                frame_end_relative = time.time()
                frame_total_time = frame_end_relative - frame_start_relative
                frame_times.append(frame_total_time)

                # Print detailed timing every 30 frames
                if count % 30 == 0:
                    avg_frame_time = np.mean(frame_times[-30:]) if len(frame_times) >= 30 else np.mean(frame_times)
                    print(f"\n{'='*80}")
                    print(f"EVENT TRACKING TIMING REPORT - Frame #{count}")
                    print(f"{'='*80}")
                    print(f"Absolute Time:        {frame_start_absolute.strftime('%H:%M:%S.%f')[:-3]}")
                    print(f"Time Since Start:     {frame_start_since_process:.3f}s")
                    print(f"ROI Used:             [{x_r}, {y_r}, {w_r}, {h_r}]")
                    print(f"{'-'*80}")
                    print(f"Phase Breakdown:")
                    print(f"  Data Read:          {data_read_time*1000:6.2f}ms ({data_read_time/frame_total_time*100:5.1f}%)")
                    print(f"  ROI Filter:         {roi_filter_time*1000:6.2f}ms ({roi_filter_time/frame_total_time*100:5.1f}%)")
                    print(f"  Frame Generation:   {frame_gen_time*1000:6.2f}ms ({frame_gen_time/frame_total_time*100:5.1f}%)")
                    print(f"  Preprocessing:      {preprocess_time*1000:6.2f}ms ({preprocess_time/frame_total_time*100:5.1f}%)")
                    print(f"  DBSCAN Clustering:  {dbscan_time*1000:6.2f}ms ({dbscan_time/frame_total_time*100:5.1f}%)")
                    print(f"  Display:            {display_time*1000:6.2f}ms ({display_time/frame_total_time*100:5.1f}%)")
                    print(f"{'-'*80}")
                    print(f"Total Frame Time:     {frame_total_time*1000:6.2f}ms")
                    print(f"Avg Frame Time (30):  {avg_frame_time*1000:6.2f}ms")
                    print(f"Estimated FPS:        {1.0/avg_frame_time:6.2f}")
                    print(f"{'='*80}\n")

                count += 1

            except Exception as e:
                print(f"EVENT TRACKING: Error in tracking loop: {e}")
                import traceback
                traceback.print_exc()
                continue

def event_tracking_3D_direct(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,event_data_version,flir_data_version,system_state, stop_event):
    """
    Direct event clustering without converting to image first.
    Clusters events directly using their (x, y) coordinates, ignoring polarity.
    """
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

    ball_pixel_3D_memory,ball_pixel_3D_dtype,ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    # Create numpy array view ONCE outside the loop
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)

    # Create array views ONCE outside the loop to avoid memory issues
    event_array_view = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
    flir_array_view = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
    ROI_array_view = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

    # Get tracking parameters from config
    tracking = Config.get_tracking()
    print(tracking)
    dbscan = DBSCAN(eps=tracking['dbscan_eps'], min_samples=tracking['dbscan_min_samples'])

    count=0

    # Track last processed data versions to skip stale data
    last_event_version = -1

    # Timing statistics
    from datetime import datetime
    process_start_time = time.time()
    frame_times = []

    try:
        while system_state.value == 'w':
            continue
        print("Start Direct Event Tracking")
        while not stop_event.is_set():
            if system_state.value == 'e':
                break

            if system_state.value == 't':
                # Check if we have new data, skip if stale
                current_event_version = event_data_version.value
                if count % 30 == 0:  # Print version info every 30 frames
                    print(f"Version check: current={current_event_version}, last={last_event_version}")
                if current_event_version == last_event_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_event_version = current_event_version

                # === Timing Section Start ===
                frame_start_absolute = datetime.now()
                frame_start_relative = time.time()
                frame_start_since_process = frame_start_relative - process_start_time

                # try:
                # Safely read event data with bounds check
                data_read_start = time.time()
                current_num_events = num_events.value
                print(f"Frame {count}: num_events = {current_num_events}")
                if current_num_events <= 0 or current_num_events > event_shape[0]:
                    print(f"Event tracker: skipping due to invalid num_events: {current_num_events}")
                    continue
                # Use pre-created views instead of creating new ones in loop
                raw_event_data = event_array_view[:current_num_events].copy()
                ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                data_read_time = time.time() - data_read_start

                # ROI filtering
                roi_filter_start = time.time()
                x_r, y_r, w_r, h_r = ROI_data
                event_data = ROI_filter(raw_event_data, ROI_data)
                print(f"Frame {count}: after ROI filter, event_data length = {len(event_data)}, raw_events = {current_num_events}")

                if len(event_data) == 0:
                    print(f"Event tracker: skipping due to empty event_data after ROI filter, raw events: {current_num_events}")
                    continue

                t = event_data['t'][-1] / 1000000
                roi_filter_time = time.time() - roi_filter_start

                # Direct clustering on event coordinates without time information
                # Extract unique (y, x) coordinates, merging all events at same position
                coordinate_extract_start = time.time()
                if len(event_data) > 0:
                    # Extract (y, x) pairs and find unique positions
                    # This merges all events at the same spatial location regardless of t and p
                    xy_coords = np.column_stack((event_data['y'], event_data['x']))
                    points = np.unique(xy_coords, axis=0)
                else:
                    points = np.empty((0, 2))

                if points.shape[0] == 0:
                    continue
                coordinate_extract_time = time.time() - coordinate_extract_start

                # DBSCAN clustering
                dbscan_start = time.time()
                dbscan.fit(points)
                labels = dbscan.labels_
                dbscan_time = time.time() - dbscan_start

                # Post-processing and ball detection
                postprocess_start = time.time()
                # Filter out noise points (label == -1)
                cluster_points = points[labels != -1]

                ball_detected = False
                ball_center_x, ball_center_y, ball_size = 0, 0, 0

                if len(cluster_points) > 0:
                    y = min(cluster_points[:, 0])
                    x = min(cluster_points[:, 1])
                    w = max(cluster_points[:, 0]) - y
                    h = max(cluster_points[:, 1]) - x

                    if w<120 and h<120 and y<1200:
                        ball_detected = True
                        ball_center_x = x + h / 2
                        ball_center_y = y + w / 2
                        ball_size = max(h, w)

                        ball_num = ball_observations.value
                        # Check bounds before writing to shared memory
                        if ball_num < tracking['max_ball_buffer'] - 1:
                            ball_pixel_3D[ball_num][0] = float(ball_center_x)
                            ball_pixel_3D[ball_num][1] = float(ball_center_y)
                            ball_pixel_3D[ball_num][2] = float(ball_size)
                            ball_pixel_3D[ball_num][3] = t
                            ball_observations.value = ball_num + 1
                postprocess_time = time.time() - postprocess_start

                # === Timing Section End ===
                frame_end_relative = time.time()
                frame_total_time = frame_end_relative - frame_start_relative
                frame_times.append(frame_total_time)

                # Print detailed timing every 30 frames
                if count % 30 == 0:
                    avg_frame_time = np.mean(frame_times[-30:]) if len(frame_times) >= 30 else np.mean(frame_times)
                    print(f"\n{'='*80}")
                    print(f"EVENT TRACKING DIRECT (tracking_2D) TIMING REPORT - Frame #{count}")
                    print(f"{'='*80}")
                    print(f"Absolute Time:        {frame_start_absolute.strftime('%H:%M:%S.%f')[:-3]}")
                    print(f"Time Since Start:     {frame_start_since_process:.3f}s")
                    print(f"ROI Used:             [{x_r}, {y_r}, {w_r}, {h_r}]")
                    print(f"Ball Detected:        {'YES' if ball_detected else 'NO'}")
                    if ball_detected:
                        print(f"Ball Position:        ({ball_center_x:.1f}, {ball_center_y:.1f}), size={ball_size:.1f}")
                    print(f"Total Observations:   {ball_observations.value}")
                    print(f"{'-'*80}")
                    print(f"Phase Breakdown:")
                    print(f"  Data Read:          {data_read_time*1000:6.2f}ms ({data_read_time/frame_total_time*100:5.1f}%)")
                    print(f"  ROI Filter:         {roi_filter_time*1000:6.2f}ms ({roi_filter_time/frame_total_time*100:5.1f}%)")
                    print(f"  Coordinate Extract: {coordinate_extract_time*1000:6.2f}ms ({coordinate_extract_time/frame_total_time*100:5.1f}%)")
                    print(f"  DBSCAN Clustering:  {dbscan_time*1000:6.2f}ms ({dbscan_time/frame_total_time*100:5.1f}%)")
                    print(f"  Post-processing:    {postprocess_time*1000:6.2f}ms ({postprocess_time/frame_total_time*100:5.1f}%)")
                    print(f"{'-'*80}")
                    print(f"Total Frame Time:     {frame_total_time*1000:6.2f}ms")
                    print(f"Avg Frame Time (30):  {avg_frame_time*1000:6.2f}ms")
                    print(f"Estimated FPS:        {1.0/avg_frame_time:6.2f}")
                    print(f"{'='*80}\n")

                count += 1
                print(count)
                # except Exception as e:
                #     # print(f"Error in direct event tracking: {e}")
                #     import traceback
                #     traceback.print_exc()
                #     continue
    finally:
        print("Direct Event tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup
