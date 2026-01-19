import cv2
import os
import sys
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
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
    Return the sub-array of `events` whose (x, y) lie inside a rectangular ROI.

    Parameters
    ----------
    events : np.ndarray
        Structured array with the dtype shown above.  Shape: (N,)
    ROI_data : tuple (x_r, y_r, w_r, h_r)
        Top-left corner (x_r, y_r) and width/height of the rectangle.

    Returns
    -------
    np.ndarray
        A **view** (no copy) of the original array containing only the
        events inside the ROI.  Use .copy() if you need an independent copy.
    """
    roi_params = Config.get_roi()
    roi_range = roi_params['roi_range']
    x_r, y_r, w_r, h_r = ROI_data
    # print(f"ROI_filter: input ROI = [{x_r}, {y_r}, {w_r}, {h_r}], input events = {len(events)}")
    if x_r == 0 and y_r == 0 and w_r == 0 and h_r == 0:
        default_bbox = Config.get_roi()['default_bbox']
        x_r, y_r, w_r, h_r = 0,0,1280,1440
        # print(f"ROI_filter: no ROI provided, using default_bbox [{x_r}, {y_r}, {w_r}, {h_r}] for performance")
    x_r = max(x_r - roi_range, 0)
    y_r = max(y_r - 2*roi_range, 0)
    w_r = w_r + 2 * roi_range
    h_r = h_r + 4 * roi_range
    mask = (
            (events['x'] >= x_r) & (events['x'] < x_r + w_r) &
            (events['y'] >= y_r) & (events['y'] < y_r + h_r)
    )

    new_event=events[mask]
    # print(f"ROI_filter: filtered events = {len(new_event)} (from {len(events)})")
    return new_event

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
    # print(tracking)
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
                # print(f"Frame {count}: num_events = {current_num_events}")
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
                # print(f"Frame {count}: after ROI filter, event_data length = {len(event_data)}, raw_events = {current_num_events}")

                if len(event_data) == 0:
                    # print(f"Event tracker: skipping due to empty event_data after ROI filter, raw events: {current_num_events}")
                    continue

                t = event_data['t'][-1] / 1000000
                roi_filter_time = time.time() - roi_filter_start

                # Direct clustering on event coordinates without time information
                coordinate_extract_start = time.time()
                if len(event_data) > 0:
                    # Method 1: Use pandas for faster unique operations (best for large datasets)
                    df = pd.DataFrame({'y': event_data['y'], 'x': event_data['x']})
                    points = df.drop_duplicates().values
                    
                    # Method 2: Use lexsort for faster unique (best balance of speed/memory)
                    # y_coords = event_data['y']
                    # x_coords = event_data['x']
                    # # Sort by x then y for lexicographic ordering
                    # sorted_idx = np.lexsort((x_coords, y_coords))
                    # y_sorted = y_coords[sorted_idx]
                    # x_sorted = x_coords[sorted_idx]
                    # # Find unique consecutive pairs
                    # unique_mask = np.concatenate(([True],
                    #                             (y_sorted[1:] != y_sorted[:-1]) |
                    #                             (x_sorted[1:] != x_sorted[:-1])))
                    # points = np.column_stack((y_sorted[unique_mask], x_sorted[unique_mask]))
                    #
                    # Method 3: Alternative - direct extraction without unique (fastest but uses more points)
                    # points = np.column_stack((event_data['y'], event_data['x']))
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

                    if w<120 or h<120 and y < 1200 :
                        ball_detected = True
                        ball_center_x = x + h / 2
                        ball_center_y = y + w / 2
                        ball_size = min(h, w)

                    ball_num = ball_observations.value
                    # Check bounds before writing to shared memory
                    if ball_num < tracking['max_ball_buffer'] - 1 and ball_detected:
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
                # print(count)
                # except Exception as e:
                #     # print(f"Error in direct event tracking: {e}")
                #     import traceback
                #     traceback.print_exc()
                #     continue
    finally:
        print("Direct Event tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup


def event_tracking_3D_direct_compare(shared_event_data, shared_flir_data, shared_ROI_data, shared_ball_pixel_3D_data,shared_ball_pixel_3D_data_raw,
                             ball_observations, num_events, event_data_version, flir_data_version, system_state,
                             stop_event):
    """
    Direct event clustering without converting to image first.
    Clusters events directly using their (x, y) coordinates, ignoring polarity.
    """
    print("PLAIN VERSION: Function started")
    import traceback
    try:
        from sklearn.cluster import DBSCAN

        event_memory, event_dtype, event_shape = shared_event_data
        event_shm = shared_memory.SharedMemory(name=event_memory.name)
        flir_memory, flir_dtype, flir_shape = shared_flir_data
        flir_shm = shared_memory.SharedMemory(name=flir_memory.name)

        print("PLAIN VERSION: Shared memory connected")

        # Get image dimensions from config
        dims = Config.get_image_dimensions()
        rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']

        # Get image processing parameters from config
        processing = Config.get_image_processing()

        ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape = shared_ball_pixel_3D_data
        ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
        # Create numpy array view ONCE outside the loop
        ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)

        ball_pixel_3D_memory_raw, ball_pixel_3D_dtype_raw, ball_pixel_3D_shape_raw = shared_ball_pixel_3D_data_raw
        ball_pixel_3D_shm_raw = shared_memory.SharedMemory(name=ball_pixel_3D_memory_raw.name)
        ball_pixel_3D_raw = np.ndarray(ball_pixel_3D_shape_raw, dtype=ball_pixel_3D_dtype_raw, buffer=ball_pixel_3D_shm_raw.buf)


        ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
        ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)

        print("PLAIN VERSION: Ball pixel and ROI memory connected")

        # Create array views ONCE outside the loop to avoid memory issues
        event_array_view = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
        flir_array_view = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
        ROI_array_view = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

        print("PLAIN VERSION: Array views created")

        # Get tracking parameters from config
        tracking = Config.get_tracking()
        print(f"PLAIN VERSION: Config loaded: {tracking}")
        dbscan = DBSCAN(eps=tracking['dbscan_eps'], min_samples=tracking['dbscan_min_samples'])
        dbscan_raw = DBSCAN(eps=tracking['dbscan_eps'], min_samples=tracking['dbscan_min_samples'])

    except Exception as e:
        print(f"PLAIN VERSION: Error during initialization: {e}")
        traceback.print_exc()
        return

    count = 0

    # Track last processed data versions to skip stale data
    last_event_version = -1

    # Timing statistics
    from datetime import datetime
    process_start_time = time.time()
    frame_times = []

    try:
        while system_state.value == 'w':
            continue
        print("Start Direct Event Tracking (PLAIN VERSION)")
        while not stop_event.is_set():
            if system_state.value == 'e':
                break

            if system_state.value == 't':
                # Check if we have new data, skip if stale
                current_event_version = event_data_version.value
                if count % 30 == 0:  # Print version info every 30 frames
                    print(f"PLAIN VERSION - Version check: current={current_event_version}, last={last_event_version}")
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
                # print(f"PLAIN VERSION - Frame {count}: num_events = {current_num_events}")
                if current_num_events <= 0 or current_num_events > event_shape[0]:
                    print(f"PLAIN VERSION - Event tracker: skipping due to invalid num_events: {current_num_events}")
                    continue
                # Use pre-created views instead of creating new ones in loop
                raw_event_data = event_array_view[:current_num_events].copy()
                event_data_raw = raw_event_data.copy()
                ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                data_read_time = time.time() - data_read_start

                # ROI filtering
                roi_filter_start = time.time()
                x_r, y_r, w_r, h_r = ROI_data
                event_data = ROI_filter(raw_event_data, ROI_data).copy()
                # print(
                #     f"Frame {count}: after ROI filter, event_data length = {len(event_data)}, raw_events = {current_num_events}")

                if len(event_data) == 0:
                    # print(
                    #     f"Event tracker: skipping due to empty event_data after ROI filter, raw events: {current_num_events}")
                    continue

                t = event_data['t'][-1] / 1000000
                roi_filter_time = time.time() - roi_filter_start

                # Direct clustering on event coordinates without time information
                coordinate_extract_start = time.time()
                if len(event_data) > 0:
                    # Method 1: Use pandas for faster unique operations (best for large datasets)
                    df = pd.DataFrame({'y': event_data['y'], 'x': event_data['x']})
                    points = df.drop_duplicates().values
                    df_raw = pd.DataFrame({'y': event_data_raw['y'], 'x': event_data_raw['x']})
                    points_raw = df_raw.drop_duplicates().values


                else:
                    points = np.empty((0, 2))
                    points_raw = np.empty((0, 2))

                if points.shape[0] == 0:
                    continue
                coordinate_extract_time = time.time() - coordinate_extract_start

                # DBSCAN clustering
                dbscan_start = time.time()
                dbscan.fit(points)
                labels = dbscan.labels_
                dbscan_time = time.time() - dbscan_start
                dbscan_raw.fit(points_raw)
                labels_raw = dbscan_raw.labels_

                # Post-processing and ball detection
                postprocess_start = time.time()
                # Filter out noise points (label == -1)
                cluster_points = points[labels != -1]
                cluster_points_raw = points_raw[labels_raw != -1]

                ball_detected = False
                ball_center_x, ball_center_y, ball_size = 0, 0, 0

                if len(cluster_points) > 0:
                    y = min(cluster_points[:, 0])
                    x = min(cluster_points[:, 1])
                    w = max(cluster_points[:, 0]) - y
                    h = max(cluster_points[:, 1]) - x

                    # if w < 120 and h < 120:
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
                if len(cluster_points_raw) > 0:
                    y_raw = min(cluster_points_raw[:, 0])
                    x_raw = min(cluster_points_raw[:, 1])
                    w_raw = max(cluster_points_raw[:, 0]) - y_raw
                    h_raw = max(cluster_points_raw[:, 1]) - x_raw

                    # if w_raw < 120 and h_raw < 120:
                    ball_detected = True
                    ball_center_x_raw = x_raw + h_raw / 2
                    ball_center_y_raw = y_raw + w_raw / 2
                    ball_size_raw = max(h_raw, w_raw)

                    ball_num_raw = ball_observations.value
                    # Check bounds before writing to shared memory
                    if ball_num_raw < tracking['max_ball_buffer'] - 1:
                        ball_pixel_3D_raw[ball_num_raw][0] = float(ball_center_x_raw)
                        ball_pixel_3D_raw[ball_num_raw][1] = float(ball_center_y_raw)
                        ball_pixel_3D_raw[ball_num_raw][2] = float(ball_size_raw)
                        ball_pixel_3D_raw[ball_num_raw][3] = t
                            # ball_observations.value = ball_num_raw + 1
                postprocess_time = time.time() - postprocess_start

                # === Timing Section End ===
                frame_end_relative = time.time()
                frame_total_time = frame_end_relative - frame_start_relative
                frame_times.append(frame_total_time)

                # Print detailed timing every 30 frames
                if count % 30 == 0:
                    avg_frame_time = np.mean(frame_times[-30:]) if len(frame_times) >= 30 else np.mean(frame_times)
                    print(f"\n{'=' * 80}")
                    print(f"EVENT TRACKING DIRECT (tracking_2D) TIMING REPORT - Frame #{count}")
                    print(f"{'=' * 80}")
                    print(f"Absolute Time:        {frame_start_absolute.strftime('%H:%M:%S.%f')[:-3]}")
                    print(f"Time Since Start:     {frame_start_since_process:.3f}s")
                    print(f"ROI Used:             [{x_r}, {y_r}, {w_r}, {h_r}]")
                    print(f"Ball Detected:        {'YES' if ball_detected else 'NO'}")
                    if ball_detected:
                        print(f"Ball Position:        ({ball_center_x:.1f}, {ball_center_y:.1f}), size={ball_size:.1f}")
                    print(f"Total Observations:   {ball_observations.value}")
                    print(f"{'-' * 80}")
                    print(f"Phase Breakdown:")
                    print(
                        f"  Data Read:          {data_read_time * 1000:6.2f}ms ({data_read_time / frame_total_time * 100:5.1f}%)")
                    print(
                        f"  ROI Filter:         {roi_filter_time * 1000:6.2f}ms ({roi_filter_time / frame_total_time * 100:5.1f}%)")
                    print(
                        f"  Coordinate Extract: {coordinate_extract_time * 1000:6.2f}ms ({coordinate_extract_time / frame_total_time * 100:5.1f}%)")
                    print(
                        f"  DBSCAN Clustering:  {dbscan_time * 1000:6.2f}ms ({dbscan_time / frame_total_time * 100:5.1f}%)")
                    print(
                        f"  Post-processing:    {postprocess_time * 1000:6.2f}ms ({postprocess_time / frame_total_time * 100:5.1f}%)")
                    print(f"{'-' * 80}")
                    print(f"Total Frame Time:     {frame_total_time * 1000:6.2f}ms")
                    print(f"Avg Frame Time (30):  {avg_frame_time * 1000:6.2f}ms")
                    print(f"Estimated FPS:        {1.0 / avg_frame_time:6.2f}")
                    print(f"{'=' * 80}\n")

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


