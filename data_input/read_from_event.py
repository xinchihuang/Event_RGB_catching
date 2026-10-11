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
def read_from_event(shared_event_data, system_state, num_event, event_data_version, stop_event):
    import time
    import numpy as np
    from multiprocessing import shared_memory
    from metavision_core.event_io import EventsIterator

    # ---- Configs
    data_reading = Config.get_data_reading()
    delta_t       = data_reading['delta_t']
    n_events      = data_reading['n_events']
    send_interval = float(data_reading['send_interval'])
    # ASAP 模式（send_interval<=0）：每次循环都立即发布
    asap_mode = not (send_interval > 0.0)
    hz = (1.0 / send_interval) if not asap_mode else 0.0

    # 预取对齐/尺寸配置，避免循环内重复取
    dims = Config.get_image_dimensions()
    alignment = Config.get_alignment()
    y_offset = float(alignment['y_coordinate_offset'])

    # ---- Iterator (live if input_path="")
    event_iterator = iter(EventsIterator(
        input_path="",
        mode="mixed",
        delta_t=delta_t,
        n_events=n_events
    ))

    # ---- Shared memory view
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    shm_np = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    latest_events = None
    have_latest   = False
    stream_ended  = False
    last_log      = 0.0

    t0 = time.perf_counter()
    next_deadline = (t0 + send_interval) if not asap_mode else None

    # -------- 方法A：时间预算式 drain，直到追上实时或耗尽预算 --------
    def drain_events_time_budget():
        """
        反复 next(iterator) 在有限时间预算内尽可能消费到“最新一批”。
        仅返回最新那一批（用于发布），以清空 backlog、避免延迟累积。
        """
        nonlocal latest_events, have_latest, stream_ended

        # 时间预算：ASAP 模式给个小预算（2ms），定频模式用 75% 周期
        if asap_mode:
            budget_ms = 100.0
        else:
            budget_ms = max(0.5, 0.75 * send_interval * 1000.0)

        start = time.perf_counter()
        got_any = False

        while True:
            # 时间到则停止（留点时间给发布/写共享内存）
            now = time.perf_counter()
            if (now - start) * 1000.0 >= budget_ms:
                break

            try:
                ev = next(event_iterator)   # 取一批事件
            except StopIteration:
                stream_ended = True
                break
            except Exception as e:
                # 采集异常：跳出本轮，等待下次循环再试
                print(f"Event acquisition error: {e}")
                break

            # 坐标变换（在 float 域处理，最终在 align_to_rgb 中裁剪）
            try:
                x = ev['x'].astype(np.float32, copy=False)
                y = ev['y'].astype(np.float32, copy=False)

                # Fixed camera mounting transform: (x, y) -> (y, y_offset - x).
                # Legacy alignment.apply_rotation is not a live switch.
                x_new = y
                y_new = -x + y_offset

                xa, ya = align_to_rgb({'x': x_new, 'y': y_new},
                                      rgb_width=dims['rgb_width'],
                                      rgb_height=dims['rgb_height'],
                                      event_width=dims['event_width'],
                                      event_height=dims['event_height'])

                ev2 = ev.copy()
                ev2['x'] = xa
                ev2['y'] = ya

                latest_events = ev2   # 只保留“最后一批”=最新
                have_latest   = True
                got_any       = True
            except Exception as e:
                print(f"Event processing error: {e}")
                have_latest = False
                # 出错不立即退出，继续尝试 drain 下一批
                continue

        return got_any

    try:
        while not stop_event.is_set() and not stream_ended:
            now = time.perf_counter()

            # ---------- 关键：先 drain 内部缓冲到“最新” ----------
            got_any = drain_events_time_budget()
            if stream_ended:
                break

            # 读取运行状态（Value('c') 可能是 bytes）
            is_running = getattr(system_state, "value", None)
            if isinstance(is_running, (bytes, bytearray)):
                is_running = is_running.decode('ascii', errors='ignore') if is_running else ''
            elif not isinstance(is_running, str):
                is_running = str(is_running) if is_running is not None else ''

            # 发布判定（ASAP 模式：每轮都发；定频：到点才发）
            should_publish = asap_mode or (now >= next_deadline)

            if should_publish:
                if is_running == 't' and have_latest and (latest_events is not None):
                    process_start = time.perf_counter()
                    try:
                        n_avail = latest_events.shape[0] if hasattr(latest_events, "shape") else 0
                        n_cap   = int(event_shape[0])
                        n_copy  = min(n_avail, n_cap)

                        if n_copy > 0:
                            shm_np[:n_copy] = latest_events[:n_copy]
                            num_event.value = n_copy
                            # Increment version to signal new data is available
                            event_data_version.value += 1
                        else:
                            num_event.value = 0

                        proc_ms = (time.perf_counter() - process_start) * 1000.0

                        # 定频模式推进无漂移的下一次截止时间
                        if not asap_mode:
                            missed = max(1, int((now - next_deadline) // send_interval) + 1)
                            next_deadline += missed * send_interval

                        # 节流日志
                        if now - last_log >= 10.0:
                            print(f"Event: {hz:.0f} Hz target | sent {n_copy} evts | proc {proc_ms:.2f} ms")
                            last_log = now

                        if (not asap_mode) and proc_ms > send_interval * 1000.0:
                            print(f"Warning: processing {proc_ms:.2f} ms > interval {send_interval*1000:.2f} ms")
                    except Exception as e:
                        print(f"Event shared-memory write error: {e}")
                else:
                    # 暂停：快速安排下次发布时间（定频）
                    if not asap_mode:
                        next_deadline = now + send_interval

            # 轻量休眠避免空转
            if asap_mode:
                time.sleep(0.0002)  # ASAP 模式给一个很小的让步
            else:
                # 睡到下一个发布点的一小部分，避免长时间忙等
                slack = (next_deadline - time.perf_counter()) if next_deadline else 0.0
                if slack > 0:
                    time.sleep(min(slack * 0.25, 0.005))

    finally:
        print("Event Camera: Cleaning up...")
        # 共享内存由主进程统一清理，这里不关闭
