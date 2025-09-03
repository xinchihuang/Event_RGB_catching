import cv2
from multiprocessing import Process, Array, Queue, Manager, shared_memory, Value
import numpy as np
import time
import os


def read_from_flir(shared_flir_data, system_state, stop_event):
    import PySpin
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()
    cam.BeginAcquisition()

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)

    # 时间控制参数
    send_interval = 0.02  # 发送间隔：0.02秒 (50Hz)
    last_send_time = 0
    latest_frame = None
    frame_ready = False

    print(f"FLIR Camera: Starting with {send_interval * 1000}ms send interval")

    # 连续获取最新帧的线程函数（模拟）
    def get_latest_frame():
        """获取并处理最新的相机帧"""
        nonlocal latest_frame, frame_ready

        image_result = cam.GetNextImage(1000)  # 减少超时时间以提高响应速度
        if image_result.IsIncomplete():
            return False

        # 处理图像数据
        image_data = image_result.GetNDArray()
        cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

        # 更新最新帧
        latest_frame = cv_image.copy()
        frame_ready = True

        # 释放图像结果
        image_result.Release()
        return True

    while not stop_event.is_set():
        current_time = time.perf_counter()

        # 持续获取最新帧，保持缓冲区最新
        try:
            get_latest_frame()
        except Exception as e:
            print(f"Frame acquisition error: {e}")
            continue

        # 检查是否到了发送时间
        if current_time - last_send_time >= send_interval:
            if system_state.value == 't' and frame_ready and latest_frame is not None:
                process_start = time.perf_counter()
                print(process_start)

                try:
                    # 发送最新帧到共享内存
                    shm = np.ndarray(latest_frame.shape, dtype=flir_dtype, buffer=flir_shm.buf)
                    shm[:] = latest_frame

                    # 计算处理时间
                    process_time = time.perf_counter() - process_start

                    # 更新发送时间
                    last_send_time = current_time

                    # 如果处理时间超过间隔，跳过下一个周期
                    if process_time > send_interval:
                        print(
                            f"Warning: Processing time {process_time * 1000:.1f}ms > interval {send_interval * 1000}ms, skipping next cycle")
                        last_send_time += send_interval  # 跳过一个周期

                    # 打印统计信息（可选，用于调试）
                    if int(current_time) % 5 == 0:  # 每5秒打印一次
                        print(f"FLIR: Send rate: {1 / send_interval:.1f}Hz, Process time: {process_time * 1000:.1f}ms")

                except Exception as e:
                    print(f"Shared memory write error: {e}")
            else:
                # 即使不发送数据，也要更新时间以保持同步
                last_send_time = current_time

        # 短暂休眠以避免过度占用CPU
        time.sleep(0.001)  # 1ms休眠

    # 清理资源
    print("FLIR Camera: Cleaning up...")
    try:
        cam.EndAcquisition()
        cam.DeInit()
        cam_list.Clear()
        system.ReleaseInstance()
    except Exception as e:
        print(f"Cleanup error: {e}")