import cv2
import os
import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import math
def read_from_video(shared_flir_data, system_state, raw_count,
                    record_index, start_ts, stop_event,video_broadcasting,
                    record_dir='/home/xinchi/Event_RGB_catching_data/saved_data_l'):

    import os, time, math, cv2
    from multiprocessing import shared_memory
    import numpy as np

    count = 0
    initialized = False
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # ---------- 准备阶段：只在 state=='w' 且未初始化时执行 ----------
        if system_state.value == 'w' and not initialized:
            video_path = os.path.join(record_dir, f"rgb/{record_index.value}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 等待 raw 线程把 raw_count 写好，否则 ratio 计算错误  ### FIX-1
            while raw_count.value < total_frames and not stop_event.is_set():
                time.sleep(0.001)

            ratio = (raw_count.value - 1) / (total_frames - 1)

            period_raw = 0.002
            period_video = ratio * period_raw
            print(f"Frame Interval: {period_video:.4f} seconds")
            print(f"Total Frames: {total_frames}, Raw Count: {raw_count.value}, Ratio: {ratio:.4f}")

            # 预读到内存
            rgb_buffer = []
            ret, frame = cap.read()
            while ret:
                rgb_buffer.append(frame)
                ret, frame = cap.read()
            print("Video ready")

            count = 0
            initialized = True

        # ---------- 播放阶段 -----------------------------------------
        if system_state.value == 't' and initialized:
            if start_ts.value == 0.0:                       # 第一次写入壁钟
                start_ts.value = time.perf_counter()

            loop_idx = count
            target_time = start_ts.value + loop_idx * period_video

            if count >= len(rgb_buffer):                   # 播完
                video_broadcasting.value = 'f'
                initialized = False                        # 只在真正结束时清零 ### FIX-2
                print("End of video data.")
                continue

            cv_image = rgb_buffer[count]

            # 写共享内存
            shm = np.ndarray(cv_image.shape, dtype=flir_dtype, buffer=flir_shm.buf)
            shm[:] = cv_image

            # 睡到周期边界
            sleep_time = target_time + period_video - time.perf_counter()  ### FIX-3
            if sleep_time > 0:
                time.sleep(sleep_time)

            count += 1

