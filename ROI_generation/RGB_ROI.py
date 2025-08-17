import cv2

from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
def pure_RGB(shared_event_data, shared_flir_data,shared_ROI_data,num_event,system_state, stop_event):

    rgb_width, rgb_height = 1080, 1440
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
    # time.sleep(2)
    # system_state.value = 't'

    count = 1

    while not stop_event.is_set():
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
        if num_event.value == 0:
            continue
        if system_state.value == 't':
            # print(count)
            start_time = time.time()
            # Convert from BGR to HSV color space

            rgb_image=flir_data

            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            lower_hsv = np.array([30, 90, 0])  # H,S,V
            upper_hsv = np.array([50, 255, 255])

            # 直接在 BGR 上 inRange
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Find the largest contour by area
                c = max(contours, key=cv2.contourArea)
                # Get bounding box coordinates from the largest contour
                x, y, w, h = cv2.boundingRect(c)
                state=[x,y,w,h]
                x_r, y_r, w_r, h_r = state
                ROI[0] = x_r
                ROI[1] = y_r
                ROI[2] = w_r
                ROI[3] = h_r
                rgb_image = flir_data
                cv2.rectangle(rgb_image, (x_r, y_r), (x_r + w_r, y_r+ h_r), (0, 255, 0), 2)

            combined = rgb_image
            scale = 0.8
            # 获取新的尺寸
            width = int(combined.shape[1] * scale)
            height = int(combined.shape[0] * scale)
            new_size = (width, height)

            resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
            end_t = time.time()
            count += 1