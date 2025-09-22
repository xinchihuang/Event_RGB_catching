import numpy as np
import time
import os
import sys
from multiprocessing import shared_memory

# Import configuration
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
def read_from_event(shared_event_data, system_state, num_event, stop_event):
    import time
    import numpy as np
    from multiprocessing import shared_memory
    from metavision_core.event_io import EventsIterator

    # ---- Configs
    data_reading = Config.get_data_reading()
    delta_t     = data_reading['delta_t']
    n_events    = data_reading['n_events']
    send_interval = float(data_reading['send_interval'])
    hz = (1.0 / send_interval) if send_interval > 0 else 0.0

    # ---- Iterator (live if input_path="")
    event_iterator = iter(EventsIterator(
        input_path="",
        mode="mixed",
        delta_t=delta_t,
        n_events=n_events
    ))

    # ---- Shared memory view (created once)
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    shm_np = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    print(f"Event Camera: send_interval={send_interval*1000:.1f} ms (~{hz:.1f} Hz)")

    latest_events = None
    have_latest = False

    # logging throttle
    last_log = 0.0

    # scheduler: drift-free deadlines
    t0 = time.perf_counter()
    next_deadline = t0 + send_interval

    def get_latest_events():
        """Pull one batch, align coords, store as latest."""
        nonlocal latest_events, have_latest
        try:
            ev = next(event_iterator)  # structured np array (x,y,p,t)
        except StopIteration:
            print("Event stream ended.")
            return False
        except Exception as e:
            print(f"Event acquisition error: {e}")
            return True  # keep trying

        try:
            # xy transform (copy to avoid aliasing)
            x = ev['x'].copy()
            y = ev['y'].copy()

            # user transform
            x_new, y_new = y, -x
            alignment = Config.get_alignment()
            y_new = y_new + alignment['y_coordinate_offset']

            # align to RGB (assumed to return (x_aligned, y_aligned))
            x_aligned, y_aligned = align_to_rgb({'x': x_new, 'y': y_new})
            # write back into a copy of ev to keep dtype/fields
            ev2 = ev.copy()
            ev2['x'] = x_aligned
            ev2['y'] = y_aligned

            latest_events = ev2
            have_latest = True
        except Exception as e:
            print(f"Event processing error: {e}")
            have_latest = False
        return True

    try:
        while not stop_event.is_set():
            now = time.perf_counter()

            # Always drain one chunk if available (keeps freshest batch)
            if not get_latest_events():
                break  # StopIteration from file source

            # Only publish at/after deadline
            if now >= next_deadline:
                # If paused, do not advance deadline; try again immediately when resumed
                if getattr(system_state, "value", None) == 't' and have_latest and latest_events is not None:
                    process_start = time.perf_counter()
                    try:
                        # Copy as many events as fit
                        # Accept both 1D (N,) and 2D (N,fields) buffers
                        n_avail = getattr(latest_events, "shape", (0,))[0] if hasattr(latest_events, "shape") else 0
                        n_cap = int(event_shape[0])
                        n_copy = min(n_avail, n_cap)

                        if n_copy > 0:
                            # zero only the portion we write to avoid stale tail when n_copy < n_cap
                            # if your consumer relies on num_event, tail won't be read anyway
                            shm_np[:n_copy] = latest_events[:n_copy]
                            num_event.value = n_copy
                        else:
                            num_event.value = 0  # explicit empty

                        proc_ms = (time.perf_counter() - process_start) * 1000.0

                        # schedule next deadline(s) without drift
                        missed = max(1, int((now - next_deadline) // send_interval) + 1)
                        next_deadline += missed * send_interval

                        # throttled logging
                        if now - last_log >= 10.0:
                            print(f"Event: {hz:.0f} Hz target | sent {n_copy} evts | proc {proc_ms:.2f} ms")
                            last_log = now

                        # overrun info
                        if proc_ms > send_interval * 1000.0:
                            print(f"Warning: processing {proc_ms:.2f} ms > interval {send_interval*1000:.2f} ms")
                    except Exception as e:
                        print(f"Event shared-memory write error: {e}")
                        # keep running; try again next slot
                else:
                    # paused: don’t move deadline forward to avoid accumulating delay on resume
                    # but do set it to now + interval so resume sends promptly
                    next_deadline = now + send_interval

            # tiny sleep to yield CPU
            time.sleep(0.0001)

    finally:
        print("Event Camera: Cleaning up...")
        try:
            # If EventsIterator exposes close/.__del__, let GC handle; otherwise ignore
            pass
        except Exception as e:
            print(f"Event iterator cleanup error: {e}")
        try:
            event_shm.close()
        except Exception as e:
            print(f"SharedMemory close error: {e}")

# def read_from_event(shared_event_data, system_state,num_event, stop_event):
#     from metavision_core.event_io import EventsIterator
#
#     # Get data reading parameters from config
#     data_reading = Config.get_data_reading()
#
#     # 初始化event迭代器
#     event_iterator = iter(EventsIterator(input_path="", mode="mixed",
#                                         delta_t=data_reading['delta_t'],
#                                         n_events=data_reading['n_events']))
#
#     # 时间控制参数
#     send_interval = data_reading['send_interval']  # 发送间隔
#     last_send_time = 0
#     latest_events = None
#     events_ready = False
#
#     # 获取共享内存信息
#     event_memory, event_dtype, event_shape = shared_event_data
#     event_shm = shared_memory.SharedMemory(name=event_memory.name)
#
#     print(f"Event Camera: Starting with {send_interval*1000}ms send interval (500Hz)")
#
#     def get_latest_events():
#         """
#         获取并处理最新的event数据
#         """
#         nonlocal latest_events, events_ready
#
#         try:
#             # 获取下一批event数据
#             event_data = next(event_iterator)
#
#             # 坐标转换处理
#             event_data['x'], event_data['y'] = event_data['y'].copy(), -event_data['x'].copy()
#             alignment = Config.get_alignment()
#             event_data['y'] += alignment['y_coordinate_offset']
#             event_data['x'], event_data['y'] = align_to_rgb(event_data)
#
#             # 更新最新events
#             latest_events = event_data.copy()
#             events_ready = True
#
#             return True
#
#         except StopIteration:
#             print("Event stream ended")
#             return False
#         except Exception as e:
#             print(f"Event acquisition error: {e}")
#             return False
#
#     while not stop_event.is_set():
#         current_time = time.perf_counter()
#
#         # 持续获取最新event数据，保持缓冲区最新
#         try:
#             if not get_latest_events():
#                 # 如果没有更多数据，退出循环
#                 break
#         except Exception as e:
#             print(f"Event processing error: {e}")
#             continue
#
#         # 检查是否到了发送时间
#         if current_time - last_send_time >= send_interval:
#             if system_state.value == 't' and events_ready and latest_events is not None and hasattr(latest_events, 'shape'):
#                 # 记录处理开始时间
#                 process_start = time.perf_counter()
#
#                 try:
#                     # 发送最新event数据到共享内存
#                     shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
#
#                     # 清空共享内存
#                     shm[:] = np.zeros(event_shape, dtype=event_dtype)
#
#                     # 复制最新event数据
#                     num_events = 0
#                     if latest_events is not None:
#                         try:
#                             num_events = min(latest_events.shape[0], event_shape[0])
#                             if num_events > 0:
#                                 shm[:num_events] = latest_events[:num_events]
#                         except (AttributeError, IndexError) as e:
#                             print(f"Event data format error: {e}")
#                             num_events = 0
#                         num_event.value = num_events
#
#                     # 计算处理时间
#                     process_time = time.perf_counter() - process_start
#
#                     # 更新发送时间
#                     last_send_time = current_time
#
#                     # 如果处理时间超过间隔，跳过下一个周期
#                     if process_time > send_interval:
#                         print(f"Warning: Event processing time {process_time*1000:.2f}ms > interval {send_interval*1000}ms, skipping next cycle")
#                         last_send_time += send_interval  # 跳过一个周期
#
#                     # 打印统计信息（可选，用于调试）
#                     if int(current_time) % 10 == 0:  # 每10秒打印一次
#                         print(f"Event: Send rate: {1/send_interval:.0f}Hz, Events: {num_events}, Process time: {process_time*1000:.2f}ms")
#
#                 except Exception as e:
#                     print(f"Event shared memory write error: {e}")
#             else:
#                 # 即使不发送数据，也要更新时间以保持同步
#                 last_send_time = current_time
#         else:
#             shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
#             shm[:] = np.zeros(event_shape, dtype=event_dtype)
#         # 短暂休眠以避免过度占用CPU
#         time.sleep(0.0001)  # 0.1ms休眠
#
#
#     # 清理资源
#     print("Event Camera: Cleaning up...")
#     try:
#         # 这里可以添加event相机的清理代码
#         pass
#     except Exception as e:
#         print(f"Event cleanup error: {e}")