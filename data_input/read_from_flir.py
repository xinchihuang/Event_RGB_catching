import cv2
from multiprocessing import Process, Array, Queue, Manager, shared_memory, Value
import numpy as np
import time
import os

def read_from_flir(shared_flir_data, flir_data_version, system_state, stop_event):
    import time
    import numpy as np
    import cv2
    from multiprocessing import shared_memory
    import PySpin

    # ---------- Camera setup ----------
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() < 1:
        system.ReleaseInstance()
        raise RuntimeError("No FLIR cameras found.")

    cam = cam_list[0]
    cam.Init()

    # Set continuous acquisition & minimize lag
    nodemap = cam.GetNodeMap()
    s_node = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if PySpin.IsAvailable(s_node) and PySpin.IsWritable(s_node):
        cont = s_node.GetEntryByName('Continuous')
        s_node.SetIntValue(cont.GetValue())

    # Prefer dropping old frames to keep freshest
    s_node_tl = cam.GetTLStreamNodeMap()
    bhm = PySpin.CEnumerationPtr(s_node_tl.GetNode('StreamBufferHandlingMode'))
    if PySpin.IsAvailable(bhm) and PySpin.IsWritable(bhm):
        newest_only = bhm.GetEntryByName('NewestOnly')
        if newest_only is not None:
            bhm.SetIntValue(newest_only.GetValue())

    # Color processor (use PySpin, faster & Bayer-safe)
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    cam.BeginAcquisition()

    # ---------- Shared memory ----------
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    shm = shared_memory.SharedMemory(name=flir_memory.name)
    out_np = np.ndarray(flir_shape, dtype=flir_dtype, buffer=shm.buf)

    # Sanity: print shape/dtype expectations once
    print(f"FLIR Camera: shared buffer shape={flir_shape}, dtype={flir_dtype}")

    # ---------- Timing ----------
    send_interval = 1.0 / 50.0  # 50 Hz
    target_hz = 1.0 / send_interval
    print(f"FLIR Camera: send_interval={send_interval*1000:.1f} ms (~{target_hz:.1f} Hz)")

    t0 = time.perf_counter()
    next_deadline = t0 + send_interval
    last_log = t0

    # ---------- Main loop ----------
    try:
        while not stop_event.is_set():
            now = time.perf_counter()

            # Get one frame (short timeout keeps loop responsive)
            image_result = cam.GetNextImage(1000)  # ms
            if image_result.IsIncomplete():
                # Always release even if incomplete
                try:
                    image_result.Release()
                except Exception:
                    pass
                # don’t block; try again
                time.sleep(0.0005)
                continue

            try:
                # Convert to BGR8 using PySpin (then rotate via OpenCV if needed)
                img_conv = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                bgr = img_conv.GetNDArray()

                # If your camera is rotated physically, keep this; else remove
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

                # Publish only on/after deadline and when running
                if now >= next_deadline and getattr(system_state, "value", None) == 't':
                    start = time.perf_counter()

                    # Validate dtype/shape match; reshape or convert as needed
                    if bgr.dtype != out_np.dtype:
                        # Convert to target dtype safely
                        bgr_send = bgr.astype(out_np.dtype, copy=False)
                    else:
                        bgr_send = bgr

                    if bgr_send.shape != out_np.shape:
                        # If your shared buffer expects a different size, resize safely
                        # NOTE: this resizes every frame; ideally set camera ROI to match flir_shape.
                        h, w = out_np.shape[:2]
                        bgr_send = cv2.resize(bgr_send, (w, h), interpolation=cv2.INTER_AREA)

                    # Copy into shared block
                    np.copyto(out_np, bgr_send, casting='unsafe')
                    # Increment version to signal new data is available
                    flir_data_version.value += 1

                    # Drift-free schedule: advance by whole intervals, skip if we overran
                    now2 = time.perf_counter()
                    missed = max(1, int((now2 - next_deadline) // send_interval) + 1)
                    next_deadline += missed * send_interval

                    proc_ms = (now2 - start) * 1000.0
                    if proc_ms > send_interval * 1000.0:
                        print(f"FLIR warn: processing {proc_ms:.2f} ms > interval {send_interval*1000:.2f} ms")

                    # Throttled stats
                    if now2 - last_log >= 10.0:
                        print(f"FLIR: {target_hz:.0f} Hz target | proc {proc_ms:.2f} ms | frame {bgr.shape[1]}x{bgr.shape[0]}")
                        last_log = now2

                elif now >= next_deadline:
                    # paused → schedule next tick to send promptly after resume
                    next_deadline = now + send_interval

            except Exception as e:
                print(f"Frame processing error: {e}")

            finally:
                # Always release the image to avoid starving the driver
                try:
                    image_result.Release()
                except Exception:
                    pass

            # Yield CPU lightly
            time.sleep(0.0005)

    finally:
        # ---------- Cleanup ----------
        print("FLIR Camera: Cleaning up...")
        try:
            cam.EndAcquisition()
        except Exception as e:
            print(f"EndAcquisition error: {e}")
        try:
            cam.DeInit()
        except Exception as e:
            print(f"DeInit error: {e}")
        try:
            cam_list.Clear()
        except Exception as e:
            print(f"Camera list clear error: {e}")
        try:
            system.ReleaseInstance()
        except Exception as e:
            print(f"System release error: {e}")
        # Do NOT close shared memory in subprocess - main process handles cleanup

# def read_from_flir(shared_flir_data, system_state, stop_event):
#     import PySpin
#     system = PySpin.System.GetInstance()
#     cam_list = system.GetCameras()
#     cam = cam_list[0]
#     cam.Init()
#     cam.BeginAcquisition()
#
#     flir_memory, flir_dtype, flir_shape = shared_flir_data
#     flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
#
#     # 时间控制参数
#     send_interval = 0.02  # 发送间隔：0.02秒 (50Hz)
#     last_send_time = 0
#     latest_frame = None
#     frame_ready = False
#
#     print(f"FLIR Camera: Starting with {send_interval * 1000}ms send interval")
#
#     # 连续获取最新帧的线程函数（模拟）
#     def get_latest_frame():
#         """获取并处理最新的相机帧"""
#         nonlocal latest_frame, frame_ready
#
#         image_result = cam.GetNextImage(1000)  # 减少超时时间以提高响应速度
#         if image_result.IsIncomplete():
#             return False
#
#         # 处理图像数据
#         image_data = image_result.GetNDArray()
#         cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
#         cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
#
#         # 更新最新帧
#         latest_frame = cv_image.copy()
#         frame_ready = True
#
#         # 释放图像结果
#         image_result.Release()
#         return True
#
#     while not stop_event.is_set():
#         current_time = time.perf_counter()
#
#         # 持续获取最新帧，保持缓冲区最新
#         try:
#             get_latest_frame()
#         except Exception as e:
#             print(f"Frame acquisition error: {e}")
#             continue
#
#         # 检查是否到了发送时间
#         if current_time - last_send_time >= send_interval:
#             if system_state.value == 't' and frame_ready and latest_frame is not None:
#                 process_start = time.perf_counter()
#                 # print(process_start)
#
#                 try:
#                     # 发送最新帧到共享内存
#                     shm = np.ndarray(latest_frame.shape, dtype=flir_dtype, buffer=flir_shm.buf)
#                     shm[:] = latest_frame
#
#                     # 计算处理时间
#                     process_time = time.perf_counter() - process_start
#
#                     # 更新发送时间
#                     last_send_time = current_time
#
#                     # 如果处理时间超过间隔，跳过下一个周期
#                     if process_time > send_interval:
#                         print(
#                             f"Warning: Processing time {process_time * 1000:.1f}ms > interval {send_interval * 1000}ms, skipping next cycle")
#                         last_send_time += send_interval  # 跳过一个周期
#
#                     # 打印统计信息（可选，用于调试）
#                     # if int(current_time) % 5 == 0:  # 每5秒打印一次
#                     #     print(f"FLIR: Send rate: {1 / send_interval:.1f}Hz, Process time: {process_time * 1000:.1f}ms")
#
#                 except Exception as e:
#                     print(f"Shared memory write error: {e}")
#             else:
#                 # 即使不发送数据，也要更新时间以保持同步
#                 last_send_time = current_time
#
#         # 短暂休眠以避免过度占用CPU
#         time.sleep(0.001)  # 1ms休眠
#
#     # 清理资源
#     print("FLIR Camera: Cleaning up...")
#     try:
#         cam.EndAcquisition()
#         cam.DeInit()
#         cam_list.Clear()
#         system.ReleaseInstance()
#     except Exception as e:
#         print(f"Cleanup error: {e}")