import cv2
import os
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import pandas as pd

# Import configuration
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config

def align_to_rgb(event_data, rgb_width=None, rgb_height=None, event_width=None, event_height=None):
    # Get dimensions from config if not provided
    if rgb_width is None or rgb_height is None or event_width is None or event_height is None:
        dims = Config.get_image_dimensions()
        rgb_width = dims['rgb_width'] if rgb_width is None else rgb_width
        rgb_height = dims['rgb_height'] if rgb_height is None else rgb_height
        event_width = dims['event_width'] if event_width is None else event_width
        event_height = dims['event_height'] if event_height is None else event_height
    
    # Get alignment parameters from config
    alignment = Config.get_alignment()
    event_pixel = alignment['event_pixel_size']
    rgb_pixel = alignment['rgb_pixel_size']
    offset_x = alignment['offset_x']
    offset_y = alignment['offset_y']
    
    x_e, y_e = event_data['x'].copy(), event_data['y'].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2

    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + offset_x
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + offset_y
    y = np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)
def read_from_raw(shared_event_data, event_data_version, system_state, num_event,
                  raw_count, record_index, start_ts, stop_event,event_broadcasting,
                  record_dir='', event_frame_offset=0, playback_speed=1.0, current_event_frame=None, is_paused=None):
    """
    播放 .raw / event 流：
    1.  先把整段 .raw 全部读取到 event_buffer（按 2 ms 一块）
    2.  播放阶段：每 2 ms
        - shm[:] = 0            # 必须整块清零（下游依赖）
        - shm[:n] = event_data  # 写本块 n 个事件
    3.  绝对时钟 = start_ts + k*period，保证与 RGB 线程同步
    """

    import os, time, numpy as np
    from multiprocessing import shared_memory
    from metavision_core.event_io import EventsIterator
    # align_to_rgb 必须在其他文件 import

    # ---------- 共享内存句柄 ----------
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    # Get data reading parameters from config
    data_reading = Config.get_data_reading()
    period = data_reading['send_interval'] / playback_speed  # Apply playback speed to period
    event_count = 0
    initialized = False

    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # ---------- 预读取 .raw ----------
        if system_state.value == 'w' and not initialized:
            raw_path = os.path.join(record_dir, f"event/{record_index.value}.raw")
            event_buffer = []
            iterator = EventsIterator(input_path=raw_path, mode="delta_t",
                                      delta_t=data_reading['delta_t'], 
                                      n_events=data_reading['n_events'])
            for ev in iterator:
                ev['x'], ev['y'] = ev['y'].copy(), -ev['x'].copy()
                alignment = Config.get_alignment()
                ev['y'] += alignment['y_coordinate_offset']
                ev['x'], ev['y'] = align_to_rgb(ev)
                event_buffer.append(ev)
            raw_count.value = len(event_buffer) # 让 RGB 线程计算 ratio
            print(f"Event chunks: {raw_count.value}")

            event_count = 0
            initialized = True
            print("Event data ready")
        # print(system_state.value)
        # ---------- 播放 ----------
        if system_state.value == 't' and initialized:
            if start_ts.value == 0.0:              # 第一次进入播放态
                start_ts.value = time.perf_counter()

            # 应用事件帧偏移量
            adjusted_event_count = event_count + event_frame_offset
            adjusted_event_count = max(0, min(adjusted_event_count, len(event_buffer) - 1))
            
            loop_idx = event_count
            target_time = start_ts.value + loop_idx * period

            if loop_idx >= len(event_buffer):
                # 播完
                event_broadcasting.value='f'
                initialized = False
                print("End of event data.")
                continue

            ev = event_buffer[adjusted_event_count]
            if ev.shape[0] == 0:
                # 没有事件，跳过
                event_count += 1
                continue
            # print(ev['t'][0])  # 打印最后一个事件的时间戳
            n = ev.shape[0]
            num_event.value = n
            # # print(n)
            # # === 必须整块清零 ===
            # shm.fill(0)            # 与 shm[:] = 0 等价，但 fill() 略快
            # === 写本块事件 ===
            shm[:min(n, 100000)] = ev[:min(n, 100000)] # 只写入 shm 的前 n 个事件
            # Increment version to signal new data is available
            event_data_version.value += 1
            if event_count % 100 == 0:  # Print every 100 frames
                print(f"Event frame {event_count}, version: {event_data_version.value}, events: {n}")
            # print(shm[:n])
            
            # Update current event frame index for debugging
            if current_event_frame is not None:
                current_event_frame.value = adjusted_event_count

            # === 到下一个周期刻度 ===
            sleep_time = target_time + period - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Check if paused and wait
            while is_paused and is_paused.value == 1 and not stop_event.is_set():
                time.sleep(0.1)  # Wait while paused

            event_count += 1
