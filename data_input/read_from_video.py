import cv2
import os
import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import math

# Import configuration
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def read_from_video(shared_flir_data, flir_data_version, system_state, raw_count,
                    record_index, start_ts, stop_event,video_broadcasting,
                    record_dir='/home/xinchi/Event_RGB_catching_data/saved_data_l', playback_speed=1.0, current_rgb_frame=None, is_paused=None):

    import os, time, math, cv2
    from multiprocessing import shared_memory
    import numpy as np

    count = 0
    initialized = False
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    # Create array view ONCE outside the loop to avoid memory issues
    flir_array_view = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # ---------- 准备阶段：只在 state=='w' 且未初始化时执行 ----------
        if system_state.value == 'w' and not initialized:
            video_path = os.path.join(record_dir, f"rgb/{record_index.value}.mp4")
            print(video_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 等待 raw 线程把 raw_count 写好，否则 ratio 计算错误  ### FIX-1
            while raw_count.value < total_frames and not stop_event.is_set():
                time.sleep(0.001)

            ratio = (raw_count.value - 1) / (total_frames - 1)
            ratio = 10
            print(f"Fixed ratio: {ratio}")
            # Get data reading parameters from config
            data_reading = Config.get_data_reading()
            period_raw = data_reading['send_interval'] / playback_speed  # Apply playback speed to period
            period_video = ratio * period_raw
            print(f"Frame Interval: {period_video:.4f} seconds")
            print(f"Total Frames: {total_frames}, Raw Count: {raw_count.value}, Ratio: {ratio:.4f}")

            # 预读到内存
            rgb_buffer = []
            ret, frame = cap.read()
            while ret:
                rgb_buffer.append(frame)
                ret, frame = cap.read()
            print(len(rgb_buffer))
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
                initialized = False                        # 只在真正结束时清零 ### FIX-
                print("End of video data.")
                continue

            cv_image = rgb_buffer[count] if count > 0 else rgb_buffer[0]
            
            # Apply brightness enhancement from config
            processing = Config.get_image_processing()
            brightness_multiplier = processing['brightness_multiplier']
            if brightness_multiplier != 1.0:
                cv_image = np.clip(cv_image.astype(np.float32) * brightness_multiplier, 0, 255).astype(np.uint8)
            
            # 写共享内存 - use pre-created view
            flir_array_view[:] = cv_image
            # Increment version to signal new data is available
            flir_data_version.value += 1
            
            # Update current RGB frame index for debugging
            if current_rgb_frame is not None:
                current_rgb_frame.value = count

            # 睡到周期边界
            sleep_time = target_time + period_video - time.perf_counter()  ### FIX-3
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Check if paused and wait
            while is_paused and is_paused.value == 1 and not stop_event.is_set():
                time.sleep(0.1)  # Wait while paused

            count += 1

