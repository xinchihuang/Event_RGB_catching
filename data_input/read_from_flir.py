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
