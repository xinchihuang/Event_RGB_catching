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
    if x_r == 0 and y_r == 0 and w_r == 0 and h_r == 0:
        return events
    x_r = max(x_r - roi_range, 0)
    y_r = max(y_r - 2*roi_range, 0)
    w_r = w_r + 2 * roi_range
    h_r = h_r + 4 * roi_range
    events_cp=events.copy()
    mask = (
            (events_cp['x'] >= x_r) & (events_cp['x'] < x_r + w_r) &
            (events_cp['y'] >= y_r) & (events_cp['y'] < y_r + h_r)
    )

    new_event=events_cp[mask]
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
    last_flir_version = -1

    try:
        while system_state.value == 'w':
            continue
        print("Start Tracking")
        while not stop_event.is_set():
            if system_state.value == 'e':
                break

            # print(system_state.value)
            if system_state.value == 't':
                # Check if we have new data, skip if stale
                current_event_version = event_data_version.value
                if current_event_version == last_event_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_event_version = current_event_version

                start_t=time.time()
                try:
                    # Safely read event data with bounds check
                    current_num_events = num_events.value
                    if current_num_events <= 0 or current_num_events > event_shape[0]:
                        continue
                    # Use pre-created views instead of creating new ones in loop
                    raw_event_data = event_array_view[:current_num_events].copy()
                    flir_data = flir_array_view.copy()
                    ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                    # print(ROI_data)
                    x_r, y_r, w_r, h_r = ROI_data
                    # ROI_data = [0, 0, 1080, 1440]
                    # ROI_data = [0, 0, 720, 1440]
                    # ROI_data = [0,0,0,0]
                    event_data = ROI_filter(raw_event_data, ROI_data)
                    # print(max(event_data['y']))
                    t = event_data['t'][-1] / 1000000
                    # print(t)
                    # Reuse pre-allocated buffer instead of creating new array
                    event_frame = event_frame_buffer
                    event_frame[:] = 0  # Clear buffer
                    event_frame_gen.generate_frame(event_data, event_frame)
                    image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(image, processing['binary_threshold'], processing['binary_max_value'], cv2.THRESH_BINARY)
                    points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
                    # if points.shape[0] == 0:
                    #     continue
                    dbscan.fit(points)
                    labels = dbscan.labels_
                    event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    cluster_points = points[labels != -1]

                    if not len(cluster_points)==0:
                        y = min(cluster_points[:, 0])
                        x = min(cluster_points[:, 1])
                        w = max(cluster_points[:, 0]) - y
                        h = max(cluster_points[:, 1]) - x
                        cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
                        # cv2.rectangle(event_image, (535,715), (545, 725), (0, 255, 0), 2)
                        # cv2.rectangle(event_image, (x_r, y_r), (x_r + h_r, y_r + w_r), (255, 255, 0), 2)
                        print("Ball detected at (x, y, w, h):", x, y, w, h)
                        if w<120 and h<120:
                            ball_num = ball_observations.value
                            # Check bounds before writing to shared memory
                            if ball_num < tracking['max_ball_buffer'] - 1:
                                ball_pixel_3D[ball_num][0] = float(x + h / 2)
                                ball_pixel_3D[ball_num][1] = float(y + h / 2)
                                ball_pixel_3D[ball_num][2] = float(h)
                                ball_pixel_3D[ball_num][3] = t
                                ball_observations.value = ball_num + 1
                            # print(ball_num)

                    rgb_image = flir_data
                    # combined = np.hstack((event_image))
                    combined = event_image
                    scale = processing['display_scale']

                    # 获取新的尺寸
                    width = int(combined.shape[1] * scale)
                    height = int(combined.shape[0] * scale)
                    new_size = (width, height)

                    # 缩放图像（使用 INTER_AREA 插值更适合缩小）
                    resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

                    # GUI display moved to dedicated gui_process.py (阶段1修改)
                    # cv2.imshow('Event Tracking', resized_img) - REMOVED
                    # cv2.waitKey(1) - REMOVED

                    count += 1
                    end_t = time.time()
                    # print("Event time:", end_t - start_t)
                except Exception as e:
                    # print(f"Error in event tracking: {e}")
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
    last_flir_version = -1

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
                if current_event_version == last_event_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_event_version = current_event_version

                start_t=time.time()
                try:
                    # Safely read event data with bounds check
                    current_num_events = num_events.value
                    if current_num_events <= 0 or current_num_events > event_shape[0]:
                        continue
                    # Use pre-created views instead of creating new ones in loop
                    raw_event_data = event_array_view[:current_num_events].copy()
                    flir_data = flir_array_view.copy()
                    ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions

                    x_r, y_r, w_r, h_r = ROI_data
                    event_data = ROI_filter(raw_event_data, ROI_data)

                    if len(event_data) == 0:
                        continue

                    t = event_data['t'][-1] / 1000000

                    # Direct clustering on event coordinates (ignoring polarity)
                    # Extract (y, x) coordinates from events
                    points = np.column_stack((event_data['y'], event_data['x']))

                    if points.shape[0] == 0:
                        continue

                    # Perform DBSCAN clustering directly on event coordinates
                    dbscan.fit(points)
                    labels = dbscan.labels_

                    # Filter out noise points (label == -1)
                    cluster_points = points[labels != -1]

                    if len(cluster_points) > 0:
                        y = min(cluster_points[:, 0])
                        x = min(cluster_points[:, 1])
                        w = max(cluster_points[:, 0]) - y
                        h = max(cluster_points[:, 1]) - x

                        print("Ball detected at (x, y, w, h):", x, y, w, h)
                        if w<120 and h<120:
                            ball_num = ball_observations.value
                            # Check bounds before writing to shared memory
                            if ball_num < tracking['max_ball_buffer'] - 1:
                                ball_pixel_3D[ball_num][0] = float(x + h / 2)
                                ball_pixel_3D[ball_num][1] = float(y + w / 2)
                                ball_pixel_3D[ball_num][2] = float(max(h, w))
                                ball_pixel_3D[ball_num][3] = t
                                ball_observations.value = ball_num + 1

                    count += 1
                    end_t = time.time()
                    # print("Direct Event time:", end_t - start_t)
                except Exception as e:
                    # print(f"Error in direct event tracking: {e}")
                    continue
    finally:
        print("Direct Event tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup