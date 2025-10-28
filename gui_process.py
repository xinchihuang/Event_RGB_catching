import cv2
import os
import sys
import numpy as np
from multiprocessing import shared_memory
import time

# Import configuration
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config

# Import event frame generation
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette


def gui_display_process(
    shared_event_data,
    shared_flir_data,
    shared_ROI_data,
    shared_ball_pixel_3D_data,
    ball_observations,
    num_event,
    event_data_version,
    flir_data_version,
    system_state,
    stop_event
):
    """
    专职GUI显示进程

    功能:
    1. 读取event数据并生成可视化帧
    2. 读取FLIR图像
    3. 读取ball_pixel_3D数据并在图像上绘制检测框
    4. 显示两个窗口: 'Event Tracking' 和 'RGB Tracking'

    注意: 这是唯一调用OpenCV GUI函数的进程
    """
    print("GUI Process: Starting...")
    sys.stdout.flush()

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']
    event_width, event_height = dims['event_width'], dims['event_height']

    # Get image processing parameters from config
    processing = Config.get_image_processing()

    # Initialize event frame generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=rgb_width,
        sensor_height=rgb_height,
        fps=processing['fps'],
        palette=ColorPalette.Dark
    )

    # 1. Open all shared memory (read-only)
    try:
        # Event data
        event_memory, event_dtype, event_shape = shared_event_data
        event_shm = shared_memory.SharedMemory(name=event_memory.name)
        event_buffer = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

        # FLIR data
        flir_memory, flir_dtype, flir_shape = shared_flir_data
        flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
        flir_buffer = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

        # ROI data
        ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
        ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
        ROI_buffer = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

        # Ball pixel 3D data
        ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape = shared_ball_pixel_3D_data
        ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
        ball_pixel_3D_buffer = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)

        print("GUI Process: Shared memory opened successfully")
        sys.stdout.flush()

    except Exception as e:
        print(f"GUI Process: Failed to open shared memory: {e}")
        sys.stdout.flush()
        return

    # 2. Initialize OpenCV window (ONLY in this process)
    cv2.startWindowThread()
    cv2.namedWindow('Event & RGB Tracking')

    print("GUI Process: Initialization complete, starting display loop...")
    sys.stdout.flush()

    # Pre-allocate buffers
    event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    event_frame = np.ascontiguousarray(event_frame)

    # Track last displayed versions to avoid displaying stale data
    last_event_version = -1
    last_flir_version = -1

    # 3. Main display loop
    while not stop_event.is_set():
        try:
            # Only display when system is in tracking mode
            if system_state.value != 't':
                time.sleep(0.01)
                continue

            # Check if we have new data, skip if stale
            current_event_version = event_data_version.value
            if current_event_version == last_event_version:
                time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                continue
            last_event_version = current_event_version

            # Read event data and generate visualization
            current_num_events = num_event.value
            if current_num_events > 0 and current_num_events <= event_shape[0]:
                event_data = event_buffer[:current_num_events].copy()

                # Generate event frame
                event_frame[:] = 0  # Clear buffer
                event_frame_gen.generate_frame(
                    event_data,
                    event_frame,
                    accumulation_time_us=processing['accumulation_time_us']
                )

                # Convert to grayscale and binary
                event_image_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(
                    event_image_gray,
                    processing['binary_threshold'],
                    processing['binary_max_value'],
                    cv2.THRESH_BINARY
                )
                event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            else:
                event_image = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)

            # Read FLIR image
            rgb_image = flir_buffer.copy()

            # Read ball_pixel_3D data and draw bounding boxes
            current_obs = ball_observations.value
            if current_obs > 0 and current_obs <= ball_pixel_3D_shape[0]:
                ball_pixels = ball_pixel_3D_buffer[:current_obs].copy()

                # Draw on event image (Event tracker detections)
                for i in range(current_obs):
                    x, y, h = ball_pixels[i][:3]
                    if h > 0:  # Valid detection
                        x_int, y_int, h_int = int(x), int(y), int(h)
                        # Draw bounding box centered at (x, y)
                        x1 = x_int - h_int // 2
                        y1 = y_int - h_int // 2
                        x2 = x_int + h_int // 2
                        y2 = y_int + h_int // 2

                        # Draw on event image
                        cv2.rectangle(event_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw on RGB image
                        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Read ROI data and draw on both images
            roi = ROI_buffer.copy()
            if roi[2] > 0 and roi[3] > 0:  # Valid ROI
                x_r, y_r, w_r, h_r = [int(v) for v in roi]
                # Draw ROI box in different color
                cv2.rectangle(event_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (255, 0, 0), 2)
                cv2.rectangle(rgb_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (255, 0, 0), 2)

            # Resize for display
            scale = processing['display_scale']

            # Resize event image
            event_height_scaled = int(event_image.shape[0] * scale)
            event_width_scaled = int(event_image.shape[1] * scale)
            event_resized = cv2.resize(event_image, (event_width_scaled, event_height_scaled), interpolation=cv2.INTER_AREA)

            # Resize RGB image
            rgb_height_scaled = int(rgb_image.shape[0] * scale)
            rgb_width_scaled = int(rgb_image.shape[1] * scale)
            rgb_resized = cv2.resize(rgb_image, (rgb_width_scaled, rgb_height_scaled), interpolation=cv2.INTER_AREA)

            # Combine images side by side (horizontally)
            # Ensure both images have the same height for concatenation
            max_height = max(event_resized.shape[0], rgb_resized.shape[0])

            # Pad images to same height if needed
            if event_resized.shape[0] < max_height:
                pad_height = max_height - event_resized.shape[0]
                event_resized = cv2.copyMakeBorder(event_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if rgb_resized.shape[0] < max_height:
                pad_height = max_height - rgb_resized.shape[0]
                rgb_resized = cv2.copyMakeBorder(rgb_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Add labels to images
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            font_color = (0, 255, 255)  # Yellow

            cv2.putText(event_resized, 'Event', (10, 30), font, font_scale, font_color, font_thickness)
            cv2.putText(rgb_resized, 'RGB', (10, 30), font, font_scale, font_color, font_thickness)

            # Concatenate horizontally
            combined_image = np.hstack([event_resized, rgb_resized])

            # Display single combined window
            cv2.imshow('Event & RGB Tracking', combined_image)
            cv2.waitKey(1)

        except Exception as e:
            print(f"GUI Process: Error in display loop: {e}")
            sys.stdout.flush()
            time.sleep(0.01)

    # 4. Cleanup: destroy windows
    print("GUI Process: Cleaning up...")
    sys.stdout.flush()

    cv2.destroyAllWindows()

    # Close shared memory
    try:
        event_shm.close()
        flir_shm.close()
        ROI_shm.close()
        ball_pixel_3D_shm.close()
    except Exception as e:
        print(f"GUI Process: Error closing shared memory: {e}")
        sys.stdout.flush()

    print("GUI Process: Exit")
    sys.stdout.flush()