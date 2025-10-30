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
    print(ROI_data)
    x_r, y_r, w_r, h_r = ROI_data
    print(f"ROI_filter: input ROI = [{x_r}, {y_r}, {w_r}, {h_r}], input events = {len(events)}")
    if x_r == 0 and y_r == 0 and w_r == 0 and h_r == 0:
        default_bbox = Config.get_roi()['default_bbox']
        x_r, y_r, w_r, h_r = 0,0,1280,1440
        print(f"ROI_filter: no ROI provided, using default_bbox [{x_r}, {y_r}, {w_r}, {h_r}] for performance")
    x_r = max(x_r - roi_range, 0)
    y_r = max(y_r - 2*roi_range, 0)
    w_r = w_r + 2 * roi_range
    h_r = h_r + 4 * roi_range
    mask = (
            (events['x'] >= x_r) & (events['x'] < x_r + w_r) &
            (events['y'] >= y_r) & (events['y'] < y_r + h_r)
    )

    new_event=events[mask]
    print(f"ROI_filter: filtered events = {len(new_event)} (from {len(events)})")
    return new_event
def event_tracking_3D(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,event_data_version,flir_data_version,system_state, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    # GUI operations moved to dedicated gui_process.py
    # cv2.startWindowThread() - REMOVED (阶段1修改)

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
    # Create numpy array view ONCE outside the loop
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)

    # Create array views ONCE outside the loop to avoid memory issues
    event_array_view = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
    flir_array_view = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
    ROI_array_view = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
    # save_dir = "/home/xinchi/Event_RGB_catching_data/frames"
    # num_file = len(
    #     [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])
    #
    # event_dir = f"{save_dir}/{num_file}/event/"
    # if not os.path.exists(event_dir):
    #     os.makedirs(event_dir)  # Corrected function name
    # bright
    # Get tracking parameters from config
    tracking = Config.get_tracking()
    print(tracking)
    dbscan = DBSCAN(eps=tracking['dbscan_eps'], min_samples=tracking['dbscan_min_samples'])
    #bright cluster
    # dbscan = DBSCAN(eps=12, min_samples=100)
    # dark

    # Pre-allocate event_frame buffer to avoid repeated allocation in loop
    event_frame_buffer = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    # dbscan = DBSCAN(eps=10, min_samples=80)
    # #dark cluster
    # dbscan = DBSCAN(eps=8, min_samples=80)
    count=0
    # time.sleep(3)
    # system_state.value='t'

    # Track last processed data versions to skip stale data
    last_event_version = -1

    # Timing statistics
    from datetime import datetime
    process_start_time = time.time()
    frame_times = []

    try:
        while system_state.value == 'w':
            continue

        while not stop_event.is_set():

            if system_state.value == 'e':
                break


            if system_state.value == 't':
                # Check if we have new data, skip if stale
                current_event_version = event_data_version.value
                if current_event_version == last_event_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_event_version = current_event_version

                # === Timing Section Start ===
                frame_start_absolute = datetime.now()
                frame_start_relative = time.time()
                frame_start_since_process = frame_start_relative - process_start_time

                try:
                    # Safely read event data with bounds check
                    data_read_start = time.time()
                    current_num_events = num_events.value
                    if current_num_events <= 0 or current_num_events > event_shape[0]:
                        continue
                    # Use pre-created views instead of creating new ones in loop
                    raw_event_data = event_array_view[:current_num_events].copy()
                    ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                    data_read_time = time.time() - data_read_start

                    # ROI filtering
                    roi_filter_start = time.time()
                    x_r, y_r, w_r, h_r = ROI_data
                    event_data = ROI_filter(raw_event_data, ROI_data)
                    t = event_data['t'][-1] / 1000000
                    roi_filter_time = time.time() - roi_filter_start

                    # Frame generation
                    frame_gen_start = time.time()
                    event_frame = event_frame_buffer
                    event_frame[:] = 0  # Clear buffer
                    event_frame_gen.generate_frame(event_data, event_frame)
                    frame_gen_time = time.time() - frame_gen_start

                    # Preprocessing
                    preprocess_start = time.time()
                    image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(image, processing['binary_threshold'], processing['binary_max_value'], cv2.THRESH_BINARY)
                    points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
                    preprocess_time = time.time() - preprocess_start

                    # DBSCAN clustering
                    dbscan_start = time.time()
                    dbscan.fit(points)
                    labels = dbscan.labels_
                    dbscan_time = time.time() - dbscan_start
                    # Post-processing and ball detection
                    postprocess_start = time.time()
                    event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    cluster_points = points[labels != -1]

                    ball_detected = False
                    ball_center_x, ball_center_y, ball_size = 0, 0, 0

                    if not len(cluster_points)==0:
                        y = min(cluster_points[:, 0])
                        x = min(cluster_points[:, 1])
                        w = max(cluster_points[:, 0]) - y
                        h = max(cluster_points[:, 1]) - x
                        cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)

                        if w<120 and h<120:
                            ball_detected = True
                            ball_center_x = x + h / 2
                            ball_center_y = y + h / 2
                            ball_size = h

                            ball_num = ball_observations.value
                            # Check bounds before writing to shared memory
                            if ball_num < tracking['max_ball_buffer'] - 1:
                                ball_pixel_3D[ball_num][0] = float(ball_center_x)
                                ball_pixel_3D[ball_num][1] = float(ball_center_y)
                                ball_pixel_3D[ball_num][2] = float(ball_size)
                                ball_pixel_3D[ball_num][3] = t
                                ball_observations.value = ball_num + 1
                    postprocess_time = time.time() - postprocess_start

                    # Visualization
                    visualization_start = time.time()
                    combined = event_image
                    scale = processing['display_scale']

                    # 获取新的尺寸
                    width = int(combined.shape[1] * scale)
                    height = int(combined.shape[0] * scale)
                    new_size = (width, height)

                    # 缩放图像（使用 INTER_AREA 插值更适合缩小）
                    resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
                    visualization_time = time.time() - visualization_start

                    # === Timing Section End ===
                    frame_end_relative = time.time()
                    frame_total_time = frame_end_relative - frame_start_relative
                    frame_times.append(frame_total_time)

                    # Print detailed timing every 30 frames
                    if count % 30 == 0:
                        avg_frame_time = np.mean(frame_times[-30:]) if len(frame_times) >= 30 else np.mean(frame_times)
                        print(f"\n{'='*80}")
                        print(f"EVENT TRACKING (tracking_2D) TIMING REPORT - Frame #{count}")
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
                        print(f"  Frame Generation:   {frame_gen_time*1000:6.2f}ms ({frame_gen_time/frame_total_time*100:5.1f}%)")
                        print(f"  Preprocessing:      {preprocess_time*1000:6.2f}ms ({preprocess_time/frame_total_time*100:5.1f}%)")
                        print(f"  DBSCAN Clustering:  {dbscan_time*1000:6.2f}ms ({dbscan_time/frame_total_time*100:5.1f}%)")
                        print(f"  Post-processing:    {postprocess_time*1000:6.2f}ms ({postprocess_time/frame_total_time*100:5.1f}%)")
                        print(f"  Visualization:      {visualization_time*1000:6.2f}ms ({visualization_time/frame_total_time*100:5.1f}%)")
                        print(f"{'-'*80}")
                        print(f"Total Frame Time:     {frame_total_time*1000:6.2f}ms")
                        print(f"Avg Frame Time (30):  {avg_frame_time*1000:6.2f}ms")
                        print(f"Estimated FPS:        {1.0/avg_frame_time:6.2f}")
                    count += 1

                except Exception as e:
                    # print(f"Error in event tracking: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        print("Event tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup

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

                    if w<120 and h<120:
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

def event_tracking_3D_hdbscan(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,event_data_version,flir_data_version,system_state, stop_event):
    """
    Direct event clustering using HDBSCAN algorithm.
    HDBSCAN is more efficient than DBSCAN for varying density clusters and large datasets.
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        print("ERROR: hdbscan not installed. Install with: pip install hdbscan")
        return

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
    # HDBSCAN parameters: min_cluster_size is similar to min_samples in DBSCAN
    # min_samples controls how conservative the clustering is
    hdbscan_clusterer = HDBSCAN(
        min_cluster_size=tracking.get('hdbscan_min_cluster_size', tracking['dbscan_min_samples']),
        min_samples=tracking.get('hdbscan_min_samples', 10),
        cluster_selection_epsilon=tracking.get('hdbscan_epsilon', 0.0),
        allow_single_cluster=True
    )

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
        print("Start HDBSCAN Event Tracking")
        while not stop_event.is_set():
            if system_state.value == 'e':
                break

            if system_state.value == 't':
                # Check if we have new data, skip if stale
                current_event_version = event_data_version.value
                if current_event_version == last_event_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_event_version = current_event_version

                # === Timing Section Start ===
                frame_start_absolute = datetime.now()
                frame_start_relative = time.time()
                frame_start_since_process = frame_start_relative - process_start_time

                try:
                    # Safely read event data with bounds check
                    data_read_start = time.time()
                    current_num_events = num_events.value
                    if current_num_events <= 0 or current_num_events > event_shape[0]:
                        continue
                    # Use pre-created views instead of creating new ones in loop
                    raw_event_data = event_array_view[:current_num_events].copy()
                    ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                    data_read_time = time.time() - data_read_start

                    # ROI filtering
                    roi_filter_start = time.time()
                    x_r, y_r, w_r, h_r = ROI_data
                    event_data = ROI_filter(raw_event_data, ROI_data)

                    if len(event_data) == 0:
                        continue

                    t = event_data['t'][-1] / 1000000
                    roi_filter_time = time.time() - roi_filter_start

                    # Direct clustering on event coordinates
                    # Extract (y, x) coordinates from events
                    coordinate_extract_start = time.time()
                    points = np.column_stack((event_data['y'], event_data['x']))

                    if points.shape[0] == 0:
                        continue
                    coordinate_extract_time = time.time() - coordinate_extract_start

                    # HDBSCAN clustering
                    hdbscan_start = time.time()
                    labels = hdbscan_clusterer.fit_predict(points)
                    hdbscan_time = time.time() - hdbscan_start

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

                        if w<120 and h<120:
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
                        print(f"EVENT TRACKING HDBSCAN (tracking_2D) TIMING REPORT - Frame #{count}")
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
                        print(f"  HDBSCAN Clustering: {hdbscan_time*1000:6.2f}ms ({hdbscan_time/frame_total_time*100:5.1f}%)")
                        print(f"  Post-processing:    {postprocess_time*1000:6.2f}ms ({postprocess_time/frame_total_time*100:5.1f}%)")
                        print(f"{'-'*80}")
                        print(f"Total Frame Time:     {frame_total_time*1000:6.2f}ms")
                        print(f"Avg Frame Time (30):  {avg_frame_time*1000:6.2f}ms")
                        print(f"Estimated FPS:        {1.0/avg_frame_time:6.2f}")
                        print(f"{'='*80}\n")

                    count += 1

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        print("HDBSCAN Event tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup

def event_tracking_3D_grid(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,event_data_version,flir_data_version,system_state, stop_event):
    """
    Grid-based event clustering for faster performance.
    Groups events into spatial grid cells and finds clusters by connected components.
    Much faster than DBSCAN/HDBSCAN for large event counts.
    """
    from scipy import ndimage

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

    # Grid-based parameters
    grid_cell_size = tracking.get('grid_cell_size', 5)  # pixels per grid cell
    min_events_per_cluster = tracking.get('grid_min_events', tracking['dbscan_min_samples'])

    # Pre-allocate grid buffer
    grid_height = (rgb_height + grid_cell_size - 1) // grid_cell_size
    grid_width = (rgb_width + grid_cell_size - 1) // grid_cell_size
    grid_buffer = np.zeros((grid_height, grid_width), dtype=np.uint16)

    # Structuring element for connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=bool)

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
        print("Start Grid-Based Event Tracking")
        while not stop_event.is_set():
            if system_state.value == 'e':
                break

            if system_state.value == 't':
                # Check if we have new data, skip if stale
                current_event_version = event_data_version.value
                if current_event_version == last_event_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_event_version = current_event_version

                # === Timing Section Start ===
                frame_start_absolute = datetime.now()
                frame_start_relative = time.time()
                frame_start_since_process = frame_start_relative - process_start_time

                try:
                    # Safely read event data with bounds check
                    data_read_start = time.time()
                    current_num_events = num_events.value
                    if current_num_events <= 0 or current_num_events > event_shape[0]:
                        continue
                    # Use pre-created views instead of creating new ones in loop
                    raw_event_data = event_array_view[:current_num_events].copy()
                    ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                    data_read_time = time.time() - data_read_start

                    # ROI filtering
                    roi_filter_start = time.time()
                    x_r, y_r, w_r, h_r = ROI_data
                    event_data = ROI_filter(raw_event_data, ROI_data)

                    if len(event_data) == 0:
                        continue

                    t = event_data['t'][-1] / 1000000
                    roi_filter_time = time.time() - roi_filter_start

                    # Grid-based clustering
                    grid_clustering_start = time.time()

                    # Clear grid buffer
                    grid_buffer[:] = 0

                    # Map events to grid cells
                    grid_y = event_data['y'] // grid_cell_size
                    grid_x = event_data['x'] // grid_cell_size

                    # Clip to grid boundaries
                    grid_y = np.clip(grid_y, 0, grid_height - 1)
                    grid_x = np.clip(grid_x, 0, grid_width - 1)

                    # Count events in each grid cell
                    for gy, gx in zip(grid_y, grid_x):
                        if grid_buffer[gy, gx] < 65535:  # Prevent overflow
                            grid_buffer[gy, gx] += 1

                    # Threshold: cells with enough events
                    binary_grid = grid_buffer >= (min_events_per_cluster // (grid_cell_size * grid_cell_size) + 1)

                    # Find connected components
                    labeled_grid, num_features = ndimage.label(binary_grid, structure=structure)

                    grid_clustering_time = time.time() - grid_clustering_start

                    # Post-processing and ball detection
                    postprocess_start = time.time()

                    ball_detected = False
                    ball_center_x, ball_center_y, ball_size = 0, 0, 0

                    if num_features > 0:
                        # Find the largest cluster (most likely the ball)
                        cluster_sizes = np.bincount(labeled_grid.ravel())[1:]  # Skip background (0)
                        if len(cluster_sizes) > 0:
                            largest_cluster = np.argmax(cluster_sizes) + 1

                            # Get bounding box of largest cluster in grid coordinates
                            cluster_mask = labeled_grid == largest_cluster
                            rows, cols = np.where(cluster_mask)

                            if len(rows) > 0:
                                # Convert back to pixel coordinates
                                y = rows.min() * grid_cell_size
                                x = cols.min() * grid_cell_size
                                h = (cols.max() - cols.min() + 1) * grid_cell_size
                                w = (rows.max() - rows.min() + 1) * grid_cell_size

                                if w<120 and h<120:
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
                        print(f"EVENT TRACKING GRID (tracking_2D) TIMING REPORT - Frame #{count}")
                        print(f"{'='*80}")
                        print(f"Absolute Time:        {frame_start_absolute.strftime('%H:%M:%S.%f')[:-3]}")
                        print(f"Time Since Start:     {frame_start_since_process:.3f}s")
                        print(f"ROI Used:             [{x_r}, {y_r}, {w_r}, {h_r}]")
                        print(f"Grid Size:            {grid_cell_size}px, Grid Dims: {grid_height}x{grid_width}")
                        print(f"Ball Detected:        {'YES' if ball_detected else 'NO'}")
                        if ball_detected:
                            print(f"Ball Position:        ({ball_center_x:.1f}, {ball_center_y:.1f}), size={ball_size:.1f}")
                        print(f"Total Observations:   {ball_observations.value}")
                        print(f"{'-'*80}")
                        print(f"Phase Breakdown:")
                        print(f"  Data Read:          {data_read_time*1000:6.2f}ms ({data_read_time/frame_total_time*100:5.1f}%)")
                        print(f"  ROI Filter:         {roi_filter_time*1000:6.2f}ms ({roi_filter_time/frame_total_time*100:5.1f}%)")
                        print(f"  Grid Clustering:    {grid_clustering_time*1000:6.2f}ms ({grid_clustering_time/frame_total_time*100:5.1f}%)")
                        print(f"  Post-processing:    {postprocess_time*1000:6.2f}ms ({postprocess_time/frame_total_time*100:5.1f}%)")
                        print(f"{'-'*80}")
                        print(f"Total Frame Time:     {frame_total_time*1000:6.2f}ms")
                        print(f"Avg Frame Time (30):  {avg_frame_time*1000:6.2f}ms")
                        print(f"Estimated FPS:        {1.0/avg_frame_time:6.2f}")
                        print(f"{'='*80}\n")

                    count += 1

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        print("Grid-Based Event tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup