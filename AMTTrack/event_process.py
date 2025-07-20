import cv2
import os
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
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
    roi_range=200
    x_r, y_r, w_r, h_r = ROI_data
    if x_r == 0 and y_r == 0 and w_r == 0 and h_r == 0:
        return events
    x_r = max(x_r - roi_range, 0)
    y_r = max(y_r - roi_range, 0)
    w_r = w_r + 2 * roi_range
    h_r = h_r + 2 * roi_range
    events_cp=events.copy()
    mask = (
            (events_cp['x'] >= x_r) & (events_cp['x'] < x_r + w_r) &
            (events_cp['y'] >= y_r) & (events_cp['y'] < y_r + h_r)
    )

    new_event=events_cp[mask]
    return new_event
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
                  record_dir='/home/xinchi/Event_RGB_catching_data/saved_data_l'):
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
            n = ev.shape[0]
            num_event.value = n
            # === 必须整块清零 ===
            shm.fill(0)            # 与 shm[:] = 0 等价，但 fill() 略快
            # === 写本块事件 ===
            shm[:n] = ev[:100000]

            # === 到下一个周期刻度 ===
            sleep_time = target_time + period - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            event_count += 1


def event_tracking_3D(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,system_state, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)

    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=0.0,
                                                       palette=ColorPalette.Dark)


    ball_pixel_3D_memory,ball_pixel_3D_dtype,ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    save_dir = "/home/xinchi/Event_RGB_catching_data/frames"
    num_file = len(
        [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])

    event_dir = f"{save_dir}/{num_file}/event/"
    if not os.path.exists(event_dir):
        os.makedirs(event_dir)  # Corrected function name
    dbscan = DBSCAN(eps=10, min_samples=60)
    count=0
    # time.sleep(3)
    # system_state.value='t'

    while system_state.value == 'w':
        continue
    print("start event tracking")
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
                raw_event_data = raw_event_data[:num_events.value]
                ROI_data = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                # print(ROI_data)
                x_r, y_r, w_r, h_r = ROI_data
                # ROI_data = [0, 0, 1080, 1440]
                # ROI_data = [0, 0, 720, 1440]
                event_data = ROI_filter(raw_event_data, ROI_data)
                # print(max(event_data['y']))
                t = event_data['t'][-1] / 1000000
                # # print(t)
                event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
                event_frame = np.ascontiguousarray(event_frame)
                event_frame_gen.generate_frame(event_data, event_frame)
                image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
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
                    ball_num = ball_observations.value
                    ball_pixel_3D[ball_num][0] = float(x + h / 2)
                    ball_pixel_3D[ball_num][1] = float(y + h / 2)
                    ball_pixel_3D[ball_num][2] = float(h)
                    ball_pixel_3D[ball_num][3] = t
                    if ball_num < 100:
                        ball_num += 1
                    print(ball_num)
                    ball_observations.value = ball_num

                rgb_image = flir_data
                combined = np.hstack((rgb_image, event_image))
                scale = 0.8

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