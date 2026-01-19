import cv2
import os
import sys
import numpy as np
from multiprocessing import shared_memory
import time
from datetime import datetime
import csv

# Import configuration
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config

# Import event frame generation
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette


def save_cached_data_to_files(data_cache, save_dir):
    """将缓存的数据保存到文件"""
    if not data_cache or not save_dir:
        return
    
    print(f"GUI Process: Saving {len(data_cache)} cached combined images to files...")
    sys.stdout.flush()
    
    for entry in data_cache:
        timestamp = entry['timestamp']
        combined_image = entry['combined_image']
        ball_info = entry.get('ball_info', None)
        
        try:
            # Create simple filename with ball detection info only
            if ball_info and ball_info['detected']:
                x, y, size = ball_info['x'], ball_info['y'], ball_info['size']
                combined_filename = os.path.join(save_dir, f"{timestamp}_ball_{int(x)}_{int(y)}_{int(size)}.png")
            else:
                combined_filename = os.path.join(save_dir, f"{timestamp}_no_ball.png")
            cv2.imwrite(combined_filename, combined_image)
            
            # Save ball detection metadata as a text file
            if ball_info:
                metadata_filename = combined_filename.replace('.png', '_metadata.txt')
                with open(metadata_filename, 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    if ball_info['detected']:
                        f.write(f"Ball Detected: YES\n")
                        f.write(f"Ball Center: ({ball_info['x']:.2f}, {ball_info['y']:.2f})\n")
                        f.write(f"Ball Size: {ball_info['size']:.2f}\n")
                        f.write(f"Ball Bounding Box: ({ball_info['bbox'][0]}, {ball_info['bbox'][1]}, {ball_info['bbox'][2]}, {ball_info['bbox'][3]})\n")
                    else:
                        f.write(f"Ball Detected: NO\n")
        
        except Exception as e:
            print(f"GUI Process: Error saving timestamp {timestamp}: {e}")
            sys.stdout.flush()
    
    print("GUI Process: Finished saving cached combined images to files.")
    sys.stdout.flush()


def create_csv_save_directory(base_dir="bbox_detections"):
    """创建保存CSV文件的文件夹，并确定本次运行的CSV文件名"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 查找现有的数字命名CSV文件
    existing_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and f[:-4].isdigit()]
    if existing_files:
        max_num = max([int(f[:-4]) for f in existing_files])
        next_num = max_num + 1
    else:
        next_num = 0
    
    csv_filename = os.path.join(base_dir, f"{next_num}.csv")
    return base_dir, csv_filename


def save_bbox_to_csv(bbox_data, csv_filename):
    """保存bounding box数据到指定CSV文件，追加写入（检查重复数据）"""
    if not bbox_data or not csv_filename:
        return
    
    # 检查文件是否存在，如果存在则读取最后一行进行比较
    last_row = None
    if os.path.exists(csv_filename):
        try:
            with open(csv_filename, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if rows:  # 如果文件不为空
                    last_row = rows[-1]
        except Exception as e:
            print(f"GUI Process: Error reading CSV file {csv_filename}: {e}")
            sys.stdout.flush()
    
    # 将bbox_data转换为字符串列表进行比较
    bbox_str = [str(x) for x in bbox_data]
    
    # 如果当前数据与最后一行相同，则跳过保存
    if last_row and last_row == bbox_str:
        print(f"GUI Process: Skipping duplicate bbox data {bbox_data}")
        sys.stdout.flush()
        return
    
    # 保存数据
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 只写入bounding box数据：x1, y1, x2, y2
        writer.writerow(bbox_data)
    
    print(f"GUI Process: Saved bbox {bbox_data} to {csv_filename}")
    sys.stdout.flush()


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
    stop_event,
    save_images=False,
    save_csv=False
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
    
    # Get ROI parameters from config
    roi_params = Config.get_roi()
    roi_range = roi_params['roi_range']

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

    # Create directory for saved images if saving is enabled
    save_dir = None
    image_cache = []  # Store multiple frames for better detection coverage
    max_cache_size = 10  # Keep last 10 frames
    if save_images:
        save_dir = "saved_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    # Create directory for CSV bbox saving (conditional)
    csv_filename = None
    if save_csv:
        csv_save_dir, csv_filename = create_csv_save_directory()
        print(f"GUI Process: CSV detections will be saved to {csv_filename}")
        sys.stdout.flush()
    else:
        print("GUI Process: CSV saving is disabled")
        sys.stdout.flush()

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

            # Read ball_pixel_3D data and draw bounding box for latest observation only
            current_obs = ball_observations.value
            ball_info = {'detected': False}
            
            if current_obs > 0 and current_obs <= ball_pixel_3D_shape[0]:
                ball_pixels = ball_pixel_3D_buffer[:current_obs].copy()

                # Draw only the latest observation (most recent detection)
                if current_obs > 0:
                    latest_idx = current_obs - 1  # Get the latest observation index
                    x, y, h = ball_pixels[latest_idx][:3]
                    if h > 0:  # Valid detection
                        x_int, y_int, h_int = int(x), int(y), int(h)
                        # Draw bounding box centered at (x, y)
                        x1 = x_int - h_int // 2
                        y1 = y_int - h_int // 2
                        x2 = x_int + h_int // 2
                        y2 = y_int + h_int // 2

                        # Draw on event image (green box)
                        cv2.rectangle(event_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                        
                        # Also draw on RGB image for better visibility
                        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Add center point marker
                        cv2.circle(event_image, (x_int, y_int), 5, (0, 255, 255), -1)  # Yellow center
                        cv2.circle(rgb_image, (x_int, y_int), 3, (0, 255, 255), -1)
                        
                        # Store ball info for metadata
                        ball_info = {
                            'detected': True,
                            'x': float(x),
                            'y': float(y), 
                            'size': float(h),
                            'bbox': [x1, y1, x2, y2]
                        }
                        
                        # Save bounding box to CSV (if enabled)
                        if save_csv and csv_filename:
                            save_bbox_to_csv([x1, y1, x2, y2], csv_filename)

            # Read ROI data and draw expanded ROI on both images
            roi = ROI_buffer.copy()
            roi_info = {'valid': False}
            
            if roi[2] > 0 and roi[3] > 0:  # Valid ROI
                x_r, y_r, w_r, h_r = [int(v) for v in roi]
                
                # Expand ROI according to roi_range (same logic as Event_tracker.py)
                x_r_expanded = max(x_r - roi_range, 0)
                y_r_expanded = max(y_r - 2*roi_range, 0)
                w_r_expanded = w_r + 2 * roi_range
                h_r_expanded = h_r + 4 * roi_range
                
                # Draw original ROI box in blue
                cv2.rectangle(event_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (255, 0, 0), 3)
                cv2.rectangle(rgb_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (255, 0, 0), 2)
                
                # Draw expanded ROI box in red (dashed if possible)
                cv2.rectangle(event_image, (x_r_expanded, y_r_expanded), 
                            (x_r_expanded + w_r_expanded, y_r_expanded + h_r_expanded), (0, 0, 255), 2)
                cv2.rectangle(rgb_image, (x_r_expanded, y_r_expanded), 
                            (x_r_expanded + w_r_expanded, y_r_expanded + h_r_expanded), (0, 0, 255), 2)
                
                # Store ROI info for metadata
                roi_info = {
                    'valid': True,
                    'x': x_r, 'y': y_r, 'w': w_r, 'h': h_r,
                    'expanded': [x_r_expanded, y_r_expanded, w_r_expanded, h_r_expanded]
                }

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

            # Add labels and detection info to images
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            font_color = (0, 255, 255)  # Yellow
            
            # Add main labels
            cv2.putText(event_resized, 'Event', (10, 30), font, font_scale + 0.3, font_color, font_thickness)
            cv2.putText(rgb_resized, 'RGB', (10, 30), font, font_scale + 0.3, font_color, font_thickness)
            
            # Add ball detection status
            if ball_info and ball_info['detected']:
                ball_status = f"Ball: ({ball_info['x']:.0f},{ball_info['y']:.0f}) Size:{ball_info['size']:.0f}"
                cv2.putText(event_resized, ball_status, (10, 60), font, font_scale, (0, 255, 0), font_thickness)
                cv2.putText(rgb_resized, ball_status, (10, 60), font, font_scale, (0, 255, 0), font_thickness)
            else:
                cv2.putText(event_resized, "Ball: Not Detected", (10, 60), font, font_scale, (0, 0, 255), font_thickness)
                cv2.putText(rgb_resized, "Ball: Not Detected", (10, 60), font, font_scale, (0, 0, 255), font_thickness)
            
            # Add ROI status
            if roi_info and roi_info['valid']:
                roi_status = f"ROI: {roi_info['w']}x{roi_info['h']}"
                cv2.putText(event_resized, roi_status, (10, 90), font, font_scale, (255, 0, 0), font_thickness)
                cv2.putText(rgb_resized, roi_status, (10, 90), font, font_scale, (255, 0, 0), font_thickness)
            
            # Add timestamp
            timestamp_display = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(event_resized, timestamp_display, (10, event_resized.shape[0] - 10), 
                       font, font_scale, font_color, font_thickness)
            cv2.putText(rgb_resized, timestamp_display, (10, rgb_resized.shape[0] - 10), 
                       font, font_scale, font_color, font_thickness)

            # Concatenate horizontally
            combined_image = np.hstack([event_resized, rgb_resized])

            # Display single combined window
            cv2.imshow('Event & RGB Tracking', combined_image)
            cv2.waitKey(1)
            
            # Cache multiple frames for saving (without detection boxes and text)
            if save_images and save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Create clean combined image without detection boxes and text
                # Need to get clean images before drawing detection boxes
                # Get original event image without boxes (reprocess from event_frame)
                event_image_clean = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                rgb_image_clean = flir_buffer.copy()
                
                # Resize clean images without any overlays
                event_clean = cv2.resize(event_image_clean, (event_width_scaled, event_height_scaled), interpolation=cv2.INTER_AREA)
                rgb_clean = cv2.resize(rgb_image_clean, (rgb_width_scaled, rgb_height_scaled), interpolation=cv2.INTER_AREA)
                
                # Pad to same height if needed
                if event_clean.shape[0] < max_height:
                    pad_height = max_height - event_clean.shape[0]
                    event_clean = cv2.copyMakeBorder(event_clean, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                if rgb_clean.shape[0] < max_height:
                    pad_height = max_height - rgb_clean.shape[0]
                    rgb_clean = cv2.copyMakeBorder(rgb_clean, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                
                # Combine clean images
                clean_combined_image = np.hstack([event_clean, rgb_clean])
                
                # Add to cache with rolling window
                cache_entry = {
                    'timestamp': timestamp,
                    'combined_image': clean_combined_image.copy(),
                    'ball_info': ball_info.copy() if ball_info else None
                }
                image_cache.append(cache_entry)
                
                # Keep only the most recent frames
                if len(image_cache) > max_cache_size:
                    image_cache.pop(0)  # Remove oldest frame

        except Exception as e:
            print(f"GUI Process: Error in display loop: {e}")
            sys.stdout.flush()
            time.sleep(0.01)

    # 4. Cleanup: destroy windows
    print("GUI Process: Cleaning up...")
    sys.stdout.flush()

    cv2.destroyAllWindows()

    # Save multiple cached frames before exit
    if save_images and image_cache:
        print(f"GUI Process: Saving {len(image_cache)} cached images...")
        save_cached_data_to_files(image_cache, save_dir)
    else:
        print("GUI Process: No images to save or saving disabled")

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