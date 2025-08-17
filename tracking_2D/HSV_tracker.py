import cv2
import os
import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import math
def RGB_tracking_3D(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,system_state, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    rgb_width, rgb_height = 1080, 1440

    ball_pixel_3D_memory,ball_pixel_3D_dtype,ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    count=0

    while system_state.value == 'w':
        continue
    print("start event tracking")
    p_x, p_y, p_w, p_h = 0,0,0,0
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # print(system_state.value)
        if system_state.value == 't':
            start_t=time.time()
            try:
                raw_event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

                flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
                # print(num_events.value)
                if num_events.value == 0:
                    continue
                event_data = raw_event_data[:num_events.value]
                t = event_data['t'][-1] / 1000000
                ROI_data = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                x,y,w,h= ROI_data
                rgb_image = flir_data
                if not p_x==x or not p_y==y or not p_w==w or not p_h==h:

                    cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
                    # cv2.rectangle(event_image, (535,715), (545, 725), (0, 255, 0), 2)
                    # cv2.rectangle(event_image, (x_r, y_r), (x_r + h_r, y_r + w_r), (255, 255, 0), 2)
                    ball_num = ball_observations.value
                    ball_pixel_3D[ball_num][0] = float(x + h / 2)
                    ball_pixel_3D[ball_num][1] = float(y + h / 2)
                    ball_pixel_3D[ball_num][2] = float(h)
                    ball_pixel_3D[ball_num][3] = t
                    if ball_num < 100:
                        ball_num += 1
                    print(ball_num)
                    ball_observations.value = ball_num
                    p_x, p_y, p_w, p_h = x, y, w, h

                combined = rgb_image
                scale = 0.75

                # 获取新的尺寸
                width = int(combined.shape[1] * scale)
                height = int(combined.shape[0] * scale)
                new_size = (width, height)

                # 缩放图像（使用 INTER_AREA 插值更适合缩小）
                resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('Image with Bounding Box', resized_img)
                cv2.waitKey(1)
                # cv2.imwrite(f"{save_dir}/{0}/event/" + str(count) + ".jpg", combined)
                # print("iters:",count)
                # print("ball_observations",ball_observations.value)

                count += 1
                end_t = time.time()
                # print("Event time:", end_t - start_t)
            except Exception as e:
                # print(f"Error in event tracking: {e}")
                continue