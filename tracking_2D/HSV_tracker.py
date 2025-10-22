import cv2
import os
import sys
import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import math

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def RGB_tracking_3D(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,event_data_version,flir_data_version,system_state, stop_event):
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

    # Get tracking parameters from config
    tracking = Config.get_tracking()

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
    count=0

    # Track last processed data versions to skip stale data
    last_flir_version = -1

    try:
        while system_state.value == 'w':
            continue
        print("start event tracking")
        p_x, p_y, p_w, p_h = 0,0,0,0
        while not stop_event.is_set():
            if system_state.value == 'e':
                break

            # print(system_state.value)
            if system_state.value == 't':
                # Check if we have new FLIR data, skip if stale
                current_flir_version = flir_data_version.value
                if current_flir_version == last_flir_version:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                last_flir_version = current_flir_version

                start_t=time.time()
                try:
                    # Safely read event data with bounds check
                    current_num_events = num_events.value
                    if current_num_events <= 0 or current_num_events > event_shape[0]:
                        continue
                    # Use pre-created views instead of creating new ones in loop
                    raw_event_data = event_array_view[:current_num_events].copy()
                    flir_data = flir_array_view.copy()
                    event_data = raw_event_data
                    t = event_data['t'][-1] / 1000000
                    ROI_data = ROI_array_view.copy()  # Copy to avoid race conditions
                    x,y,w,h= ROI_data
                    rgb_image = flir_data
                    if (not p_x==x or not p_y==y or not p_w==w or not p_h==h) and (h>20 or w>20):

                        cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
                        # cv2.rectangle(event_image, (535,715), (545, 725), (0, 255, 0), 2)
                        # cv2.rectangle(event_image, (x_r, y_r), (x_r + h_r, y_r + w_r), (255, 255, 0), 2)
                        ball_num = ball_observations.value
                        # Check bounds before writing to shared memory
                        if ball_num < tracking['max_ball_buffer'] - 1:
                            ball_pixel_3D[ball_num][0] = float(x + h / 2)
                            ball_pixel_3D[ball_num][1] = float(y + h / 2)
                            ball_pixel_3D[ball_num][2] = float(h)
                            ball_pixel_3D[ball_num][3] = t
                            ball_observations.value = ball_num + 1
                            print(ball_num + 1,"RGB",x,y,w,h)
                        p_x, p_y, p_w, p_h = x, y, w, h

                    combined = rgb_image
                    scale = tracking['hsv_display_scale']

                    # 获取新的尺寸
                    width = int(combined.shape[1] * scale)
                    height = int(combined.shape[0] * scale)
                    new_size = (width, height)

                    # 缩放图像（使用 INTER_AREA 插值更适合缩小）
                    resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

                    # GUI display moved to dedicated gui_process.py (阶段1修改)
                    # cv2.imshow('RGB Tracking', resized_img) - REMOVED
                    # cv2.waitKey(1) - REMOVED

                    # cv2.imwrite(f"{save_dir}/{0}/event/" + str(count) + ".jpg", combined)
                    # print("iters:",count)
                    # print("ball_observations",ball_observations.value)

                    count += 1
                    end_t = time.time()
                    # print("Event time:", end_t - start_t)
                except Exception as e:
                    # print(f"Error in event tracking: {e}")
                    continue
    finally:
        print("RGB tracking: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup