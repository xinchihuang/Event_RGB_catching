import cv2
import os
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import pandas as pd
def align_to_rgb(event_data,rgb_width=1080, rgb_height = 1440, event_width=720, event_height = 1280):
    x_e,y_e=event_data['x'].copy(),event_data['y'].copy()
    rgb_center_x=rgb_width/2
    rgb_center_y=rgb_height/2
    # print(rgb_center_x,rgb_center_y)
    event_center_x=event_width/2
    event_center_y=event_height/2
    event_pixel=3.45
    rgb_pixel=4.86

    x = (x_e-event_center_x) * rgb_pixel / event_pixel + rgb_center_x+30
    y = (y_e-event_center_y) * rgb_pixel / event_pixel + rgb_center_y+29
    y= np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)
def read_from_raw(shared_event_data, system_state, num_event,
                  raw_count, record_index, start_ts, stop_event,event_broadcasting,
                  record_dir=''):
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

    period = 0.002            # 3 ms / 块
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
                                      delta_t=2000, n_events=10000)
            for ev in iterator:
                ev['x'], ev['y'] = ev['y'].copy(), -ev['x'].copy()
                ev['y'] += 1279
                ev['x'], ev['y'] = align_to_rgb(ev)
                event_buffer.append(ev)
            mod_100= len(event_buffer) % 100
            event_buffer=event_buffer[:-mod_100] if mod_100!=0 else event_buffer
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

            loop_idx = event_count
            target_time = start_ts.value + loop_idx * period

            if loop_idx >= len(event_buffer):
                # 播完
                event_broadcasting.value='f'
                initialized = False
                print("End of event data.")
                continue

            ev = event_buffer[event_count]
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
            shm[:n] = ev[:min(n, 100000)] # 只写入 shm 的前 n 个事件
            # print(shm[:n])

            # === 到下一个周期刻度 ===
            sleep_time = target_time + period - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            event_count += 1