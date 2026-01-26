import cv2
import os
import sys
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def pure_RGB(shared_event_data, shared_flir_data,shared_ROI_data,num_event,event_data_version,flir_data_version,system_state, stop_event):

    # Initialize CV2 for multiprocessing
    cv2.startWindowThread()

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']

    # Get ROI parameters from config
    roi_params = Config.get_roi()

    # Get image processing parameters from config
    processing = Config.get_image_processing()
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

    # Create array view ONCE outside the loop to avoid memory issues
    flir_array_view = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    # Pre-allocate HSV threshold arrays outside the loop
    lower_hsv = np.array(roi_params['hsv_lower'])  # H,S,V
    upper_hsv = np.array(roi_params['hsv_upper'])

    # Track last processed data versions to skip stale data
    last_flir_version = -1

    count = 1

    try:
        while not stop_event.is_set():
            # Check if we have new FLIR data, skip if stale
            current_flir_version = flir_data_version.value
            if current_flir_version == last_flir_version:
                time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                continue
            last_flir_version = current_flir_version

            flir_data = flir_array_view.copy()
            # print(flir_data)
            # if num_event.value == 0:
            #     continue
            if system_state.value == 't':
                print(f"RGB_ROI: Processing frame, system_state={system_state.value}")
                frame_start = time.perf_counter()
                try:
                    # Convert from BGR to HSV color space

                    rgb_image = flir_data

                    # Debug: Check if flir_data is valid
                    if rgb_image is None or rgb_image.size == 0:
                        print("RGB_ROI: Warning - flir_data is empty")
                        continue

                    print(f"RGB_ROI: Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

                    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                    # lower_hsv and upper_hsv now pre-allocated outside loop

                    print(f"RGB_ROI: HSV range - lower: {lower_hsv}, upper: {upper_hsv}")

                    # 直接在 BGR 上 inRange
                    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

                    # Debug: Check mask
                    mask_pixels = np.sum(mask > 0)
                    print(f"RGB_ROI: Mask has {mask_pixels} white pixels")

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(roi_params['kernel_size']))
                    iterations = roi_params['morphology_iterations']
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

                    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    print(f"RGB_ROI: Found {len(contours)} contours")

                    if len(contours) > 0:
                        # Find the largest contour by area
                        c = max(contours, key=cv2.contourArea)
                        # Get bounding box coordinates from the largest contour
                        x, y, w, h = cv2.boundingRect(c)
                        state = [x, y, w, h]
                        x_r, y_r, w_r, h_r = state
                        if y_r > 1200:
                            continue

                        # Update ROI
                        ROI[0] = x_r
                        ROI[1] = y_r
                        ROI[2] = w_r
                        ROI[3] = h_r

                        print(f"RGB_ROI: Updated ROI to [{x_r}, {y_r}, {w_r}, {h_r}]")

                        # Draw rectangle on image copy for display
                        rgb_image = flir_data.copy()
                        cv2.rectangle(rgb_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 255, 0), 2)
                    else:
                        print("RGB_ROI: No contours found, ROI not updated")
                        rgb_image = flir_data

                    combined = rgb_image
                    scale = processing['display_scale']
                    # 获取新的尺寸
                    width = int(combined.shape[1] * scale)
                    height = int(combined.shape[0] * scale)
                    new_size = (width, height)

                    resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
                finally:
                    frame_end = time.perf_counter()
                    frame_ms = (frame_end - frame_start) * 1000.0
                    print(f"RGB_ROI: Frame {count} processing time: {frame_ms:.2f}ms")
                    count += 1
    finally:
        print("RGB ROI: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup
