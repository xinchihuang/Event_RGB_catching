import numpy as np
import time
from multiprocessing import shared_memory

def read_from_event(shared_event_data, system_state, stop_event):
    from metavision_core.event_io import EventsIterator

    # 初始化event迭代器
    event_iterator = iter(EventsIterator(input_path="", mode="mixed", delta_t=1000, n_events=10000))
    
    # 时间控制参数
    send_interval = 0.002  # 发送间隔：0.002秒 (500Hz)
    last_send_time = 0
    latest_events = None
    events_ready = False
    
    # 获取共享内存信息
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    
    print(f"Event Camera: Starting with {send_interval*1000}ms send interval (500Hz)")
    
    def get_latest_events():
        """
        获取并处理最新的event数据
        """
        nonlocal latest_events, events_ready
        
        try:
            # 获取下一批event数据
            event_data = next(event_iterator)
            
            # 坐标转换处理
            event_data['x'], event_data['y'] = event_data['y'].copy(), -event_data['x'].copy()
            event_data['y'] += 1279
            
            # 更新最新events
            latest_events = event_data.copy()
            events_ready = True
            
            return True
            
        except StopIteration:
            print("Event stream ended")
            return False
        except Exception as e:
            print(f"Event acquisition error: {e}")
            return False
    
    while not stop_event.is_set():
        current_time = time.perf_counter()
        
        # 持续获取最新event数据，保持缓冲区最新
        try:
            if not get_latest_events():
                # 如果没有更多数据，退出循环
                break
        except Exception as e:
            print(f"Event processing error: {e}")
            continue
            
        # 检查是否到了发送时间
        if current_time - last_send_time >= send_interval:
            if system_state.value == 't' and events_ready and latest_events is not None and hasattr(latest_events, 'shape'):
                # 记录处理开始时间
                process_start = time.perf_counter()
                
                try:
                    # 发送最新event数据到共享内存
                    shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
                    
                    # 清空共享内存
                    shm[:] = np.zeros(event_shape, dtype=event_dtype)
                    
                    # 复制最新event数据
                    num_events = 0
                    if latest_events is not None:
                        try:
                            num_events = min(latest_events.shape[0], event_shape[0])
                            if num_events > 0:
                                shm[:num_events] = latest_events[:num_events]
                        except (AttributeError, IndexError) as e:
                            print(f"Event data format error: {e}")
                            num_events = 0
                    
                    # 计算处理时间
                    process_time = time.perf_counter() - process_start
                    
                    # 更新发送时间
                    last_send_time = current_time
                    
                    # 如果处理时间超过间隔，跳过下一个周期
                    if process_time > send_interval:
                        print(f"Warning: Event processing time {process_time*1000:.2f}ms > interval {send_interval*1000}ms, skipping next cycle")
                        last_send_time += send_interval  # 跳过一个周期
                    
                    # 打印统计信息（可选，用于调试）
                    if int(current_time) % 10 == 0:  # 每10秒打印一次
                        print(f"Event: Send rate: {1/send_interval:.0f}Hz, Events: {num_events}, Process time: {process_time*1000:.2f}ms")
                        
                except Exception as e:
                    print(f"Event shared memory write error: {e}")
            else:
                # 即使不发送数据，也要更新时间以保持同步
                last_send_time = current_time
        
        # 短暂休眠以避免过度占用CPU
        time.sleep(0.0001)  # 0.1ms休眠
    
    # 清理资源
    print("Event Camera: Cleaning up...")
    try:
        # 这里可以添加event相机的清理代码
        pass
    except Exception as e:
        print(f"Event cleanup error: {e}")