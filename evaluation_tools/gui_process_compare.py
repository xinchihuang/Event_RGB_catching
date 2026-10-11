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


def save_cached_data_to_files(data_cache, save_dir):
    """Save buffered RGB images with boxes, event images without boxes, and detection text files."""
    if not data_cache or not save_dir:
        return
    
    print(f"GUI Process: Saving {len(data_cache)} cached images and detection results to files...")
    sys.stdout.flush()
    
    # Create subdirectories for RGB, Event images and detection results
    rgb_dir = os.path.join(save_dir, "rgb")
    event_dir = os.path.join(save_dir, "event")
    results_dir = os.path.join(save_dir, "detection_results")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create summary file for all detection results
    summary_file = os.path.join(save_dir, "detection_summary.txt")
    
    with open(summary_file, 'w') as summary_f:
        summary_f.write("Detection Results Summary\n")
        summary_f.write("========================\n")
        summary_f.write("Format: Frame_ID | Detection_Result\n")
        summary_f.write("Detection format: (x, y, size) or 'None' if not detected\n\n")
        
        for idx, entry in enumerate(data_cache):
            rgb_image_with_boxes = entry['combined_image_with_boxes']
            event_image_clean = entry['event_image_clean']
            detection_info = entry.get('detection_info', {'detected': False})
            
            try:
                # Simple filename with just sequence number
                frame_id = f"{idx:04d}"
                rgb_filename = f"{frame_id}.png"
                event_filename = f"{frame_id}.png"
                result_filename = f"{frame_id}.txt"
                
                # Save RGB image with detection boxes
                rgb_filepath = os.path.join(rgb_dir, rgb_filename)
                cv2.imwrite(rgb_filepath, rgb_image_with_boxes)
                
                # Save clean event image without boxes
                event_filepath = os.path.join(event_dir, event_filename)
                cv2.imwrite(event_filepath, event_image_clean)
                
                # Save detection results for this frame
                result_filepath = os.path.join(results_dir, result_filename)
                with open(result_filepath, 'w') as result_f:
                    result_f.write(f"Frame: {frame_id}\n")
                    result_f.write("=" * 30 + "\n")
                    
                    # Combined tracking results
                    result_f.write("Detection Result:\n")
                    if detection_info['detected']:
                        center = detection_info['center']
                        size = detection_info['size']
                        result_f.write(f"  Detected: YES\n")
                        result_f.write(f"  Center: ({center[0]:.2f}, {center[1]:.2f})\n")
                        result_f.write(f"  Size: {size:.2f}\n")
                        
                        # Calculate bbox
                        half_size = int(size / 2)
                        x_int, y_int = int(center[0]), int(center[1])
                        bbox = (x_int - half_size, y_int - half_size, x_int + half_size, y_int + half_size)
                        result_f.write(f"  BBox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})\n")
                        detection_summary = f"({center[0]:.1f}, {center[1]:.1f}, {size:.1f})"
                    else:
                        result_f.write(f"  Detected: NO\n")
                        detection_summary = "None"
                
                # Write to summary file
                summary_f.write(f"{frame_id} | {detection_summary}\n")
            
            except Exception as e:
                print(f"GUI Process: Error saving frame {idx}: {e}")
                sys.stdout.flush()
    
    print("GUI Process: Finished saving cached images and detection results to files.")
    sys.stdout.flush()


def gui_display_process_compare(
    shared_event_data,
    shared_flir_data,
    shared_ROI_data,
    shared_ball_pixel_3D_data,
    shared_ball_pixel_3D_data_rgb,
    shared_ball_pixel_3D_data_raw,
    ball_observations,
    num_event,
    event_data_version,
    flir_data_version,
    system_state,
    stop_event,
    save_images=True
):
    """
    Dedicated GUI process comparing three tracking methods.

    Responsibilities:
    1. Read event data and generate visualization frames.
    2. Read FLIR images.
    3. Read three ball_pixel_3D buffers and draw their detection boxes.
    4. Show comparison windows: RGB Tracking, Event Tracking, Pure Event Tracking.

    This is the only process that calls OpenCV GUI functions.
    """
    print("GUI Comparison Process: Starting...")
    sys.stdout.flush()

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']
    event_width, event_height = dims['event_width'], dims['event_height']

    # Get image processing parameters from config
    processing = Config.get_image_processing()
    box_thickness = 6
    
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

        # Ball pixel 3D data (event tracking)
        ball_memory, ball_dtype, ball_shape = shared_ball_pixel_3D_data
        ball_shm = shared_memory.SharedMemory(name=ball_memory.name)
        ball_buffer = np.ndarray(ball_shape, dtype=ball_dtype, buffer=ball_shm.buf)
        
        # Ball pixel 3D data RGB (RGB tracking)
        ball_rgb_memory, ball_rgb_dtype, ball_rgb_shape = shared_ball_pixel_3D_data_rgb
        ball_rgb_shm = shared_memory.SharedMemory(name=ball_rgb_memory.name)
        ball_rgb_buffer = np.ndarray(ball_rgb_shape, dtype=ball_rgb_dtype, buffer=ball_rgb_shm.buf)
        
        # Ball pixel 3D data raw (raw event tracking)
        ball_raw_memory, ball_raw_dtype, ball_raw_shape = shared_ball_pixel_3D_data_raw
        ball_raw_shm = shared_memory.SharedMemory(name=ball_raw_memory.name)
        ball_raw_buffer = np.ndarray(ball_raw_shape, dtype=ball_raw_dtype, buffer=ball_raw_shm.buf)

        print("GUI Comparison Process: Shared memory opened successfully")
        sys.stdout.flush()

    except Exception as e:
        print(f"GUI Comparison Process: Failed to open shared memory: {e}")
        sys.stdout.flush()
        return

    # 2. Initialize OpenCV window (ONLY in this process)
    cv2.startWindowThread()
    cv2.namedWindow('Tracking Comparison: RGB & Event Cameras')

    # Create directory for saved images if saving is enabled
    save_dir = None
    image_cache = []
    max_cache_size = 100  # Increased to save more images
    if save_images:
        save_dir = "saved_images_comparison"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print("GUI Comparison Process: Initialization complete, starting display loop...")
    sys.stdout.flush()

    # Pre-allocate buffers
    event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    event_frame = np.ascontiguousarray(event_frame)

    # Track last displayed versions to avoid displaying stale data
    last_event_version = -1

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
                time.sleep(0.001)
                continue
            last_event_version = current_event_version

            # Read event data and generate custom visualization
            current_num_events = num_event.value
            if current_num_events > 0 and current_num_events <= event_shape[0]:
                event_data = event_buffer[:current_num_events].copy()

                # Create custom red-blue-white event frame
                base_event_image = np.full((rgb_height, rgb_width, 3), 255, dtype=np.uint8)  # White background
                
                # Process each event and color based on polarity
                if len(event_data) > 0:
                    for event in event_data:
                        x, y, polarity, t = int(event['x']), int(event['y']), int(event['p']), int(event['t'])
                        if 0 <= x < rgb_width and 0 <= y < rgb_height:
                            if polarity == 1:  # on event
                                base_event_image[y, x] = [0, 0, 255]  # Red
                            else:  # off event (polarity == 0 or -1)
                                base_event_image[y, x] = [255, 0, 0]  # Blue
            else:
                # No events - white background
                base_event_image = np.full((rgb_height, rgb_width, 3), 255, dtype=np.uint8)

            # Read FLIR image
            base_rgb_image = flir_buffer.copy()

            # Read ROI data
            roi = ROI_buffer.copy()

            # Create separate RGB and event tracking images
            rgb_tracking_image = base_rgb_image.copy()
            event_tracking_image = base_event_image.copy()
            
            # Create a combined image with all detection boxes for saving
            combined_image_with_boxes = base_rgb_image.copy()

            # Detection info for all methods
            detection_info = {
                'event_tracking': {'detected': False},
                'rgb_tracking': {'detected': False}, 
                'raw_event_tracking': {'detected': False}
            }

            def draw_detection_from_buffer(rgb_image, combined_save_image, buffer, shape, color, method_name, info_dict):
                """Draw detections from the specified buffer."""
                try:
                    if ball_observations.value > 0:
                        latest_idx = ball_observations.value - 1
                        if latest_idx < shape[0]:
                            x, y, size, timestamp = buffer[latest_idx]
                            if x > 0 and y > 0 and size > 0:  # Valid detection
                                info_dict[method_name] = {
                                    'detected': True,
                                    'center': (float(x), float(y)),
                                    'size': float(size),
                                    'timestamp': float(timestamp)
                                }
                                
                                # Draw bounding box on RGB image
                                half_size = int(size / 2)
                                x_int, y_int = int(x), int(y)
                                cv2.rectangle(rgb_image, 
                                            (x_int - half_size, y_int - half_size),
                                            (x_int + half_size, y_int + half_size),
                                            color, box_thickness)
                                cv2.circle(rgb_image, (x_int, y_int), 3, color, -1)
                                
                                # Draw on combined save image as well
                                cv2.rectangle(combined_save_image, 
                                            (x_int - half_size, y_int - half_size),
                                            (x_int + half_size, y_int + half_size),
                                            color, box_thickness)
                                cv2.circle(combined_save_image, (x_int, y_int), 3, color, -1)
                            else:
                                info_dict[method_name]['detected'] = False
                        else:
                            info_dict[method_name]['detected'] = False
                    else:
                        info_dict[method_name]['detected'] = False
                except Exception as e:
                    print(f"Error drawing {method_name}: {e}")
                    info_dict[method_name]['detected'] = False

            # Draw detections from all three tracking methods with different colors
            # Event tracking - Red (0, 0, 255)
            draw_detection_from_buffer(rgb_tracking_image, combined_image_with_boxes, 
                                     ball_buffer, ball_shape, (0, 0, 255), 'event_tracking', detection_info)
            
            # RGB tracking - Green (0, 255, 0)
            draw_detection_from_buffer(rgb_tracking_image, combined_image_with_boxes,
                                     ball_rgb_buffer, ball_rgb_shape, (0, 255, 0), 'rgb_tracking', detection_info)
            
            # Raw event tracking - Blue (255, 0, 0)
            draw_detection_from_buffer(rgb_tracking_image, combined_image_with_boxes,
                                     ball_raw_buffer, ball_raw_shape, (255, 0, 0), 'raw_event_tracking', detection_info)

            # Resize for display
            scale = processing['display_scale']
            
            # Resize RGB tracking image
            rgb_height_scaled = int(rgb_tracking_image.shape[0] * scale)
            rgb_width_scaled = int(rgb_tracking_image.shape[1] * scale)
            rgb_resized = cv2.resize(rgb_tracking_image, (rgb_width_scaled, rgb_height_scaled), interpolation=cv2.INTER_AREA)
            
            # Resize Event tracking image
            event_height_scaled = int(event_tracking_image.shape[0] * scale)
            event_width_scaled = int(event_tracking_image.shape[1] * scale)
            event_resized = cv2.resize(event_tracking_image, (event_width_scaled, event_height_scaled), interpolation=cv2.INTER_AREA)
            
            # Ensure both images have the same height for horizontal concatenation
            max_height = max(rgb_resized.shape[0], event_resized.shape[0])
            
            # Pad to same height if needed
            if rgb_resized.shape[0] < max_height:
                pad_height = max_height - rgb_resized.shape[0]
                rgb_resized = cv2.copyMakeBorder(rgb_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if event_resized.shape[0] < max_height:
                pad_height = max_height - event_resized.shape[0]
                event_resized = cv2.copyMakeBorder(event_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # No text overlay - clean images with only detection boxes
            
            # Combine RGB and Event images side by side
            combined_image = np.hstack([rgb_resized, event_resized])

            # Display combined window
            cv2.imshow('Tracking Comparison: RGB & Event Cameras', combined_image)
            cv2.waitKey(1)
            
            # Cache frames for saving - save RGB image with boxes and clean event image
            if save_images and save_dir:
                # Resize the combined image with boxes for saving
                combined_resized = cv2.resize(combined_image_with_boxes, (rgb_width_scaled, rgb_height_scaled), interpolation=cv2.INTER_AREA)
                
                # Resize the clean event image for saving
                event_clean_resized = cv2.resize(base_event_image, (rgb_width_scaled, rgb_height_scaled), interpolation=cv2.INTER_AREA)
                
                # Add to cache with both RGB (with boxes) and Event (clean) images
                cache_entry = {
                    'combined_image_with_boxes': combined_resized.copy(),
                    'event_image_clean': event_clean_resized.copy(),
                    'detection_info': detection_info.copy() if detection_info else None
                }
                image_cache.append(cache_entry)
                
                # Keep only recent frames
                if len(image_cache) > max_cache_size:
                    image_cache.pop(0)

        except Exception as e:
            print(f"GUI Comparison Process: Error in display loop: {e}")
            sys.stdout.flush()
            time.sleep(0.01)

    # 4. Cleanup
    print("GUI Comparison Process: Cleaning up...")
    sys.stdout.flush()

    cv2.destroyAllWindows()

    # Save cached frames before exit
    if save_images and image_cache:
        print(f"GUI Comparison Process: Saving {len(image_cache)} cached images...")
        save_cached_data_to_files(image_cache, save_dir)
    else:
        print("GUI Comparison Process: No images to save or saving disabled")

    # Close shared memory
    try:
        event_shm.close()
        flir_shm.close()
        ROI_shm.close()
        ball_shm.close()
        ball_rgb_shm.close()
        ball_raw_shm.close()
    except Exception as e:
        print(f"GUI Comparison Process: Error closing shared memory: {e}")
        sys.stdout.flush()

    print("GUI Comparison Process: Exit")
    sys.stdout.flush()


def gui_display_process_compare_from_files(
    shared_event_data,
    shared_flir_data,
    num_event,
    event_data_version,
    flir_data_version,
    system_state,
    stop_event,
    results_paths,
    save_images=False
):
    """
    Playback GUI process using saved tracking results.
    Draws Model/RGB/Event boxes on both RGB and Event frames.
    """
    print("GUI Comparison Playback: Starting...")
    sys.stdout.flush()

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']

    # Get image processing parameters from config
    processing = Config.get_image_processing()
    box_thickness = 8

    # 1. Open shared memory (read-only)
    try:
        # Event data
        event_memory, event_dtype, event_shape = shared_event_data
        event_shm = shared_memory.SharedMemory(name=event_memory.name)
        event_buffer = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

        # FLIR data
        flir_memory, flir_dtype, flir_shape = shared_flir_data
        flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
        flir_buffer = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

        print("GUI Comparison Playback: Shared memory opened successfully")
        sys.stdout.flush()

    except Exception as e:
        print(f"GUI Comparison Playback: Failed to open shared memory: {e}")
        sys.stdout.flush()
        return

    def load_tracking_csv(path):
        if not path or not os.path.isfile(path):
            return np.empty((0, 4), dtype=np.float64)
        skip_rows = 0
        try:
            with open(path, "r", encoding="utf-8") as handle:
                first_line = handle.readline().strip().lower()
            if first_line and not first_line[0].isdigit() and first_line[0] not in "+-.":
                skip_rows = 1
        except Exception:
            skip_rows = 0
        data = np.loadtxt(path, delimiter=",", skiprows=skip_rows)
        if data.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        data = np.atleast_2d(data)
        return data[:, :4]

    # Load saved tracking results
    results = {}
    for key, path in (results_paths or {}).items():
        results[str(key).lower()] = load_tracking_csv(path)

    indexes = {name: -1 for name in results.keys()}
    last_event_time = None

    color_map = {
        "model": (0, 0, 255),  # Red
        "rgb": (0, 255, 0),    # Green
        "event": (255, 0, 0),  # Blue
    }

    # 2. Initialize OpenCV window
    cv2.startWindowThread()
    cv2.namedWindow('Tracking Playback: RGB & Event Cameras')

    print("GUI Comparison Playback: Initialization complete, starting display loop...")
    sys.stdout.flush()

    # Track last displayed versions to avoid displaying stale data
    last_event_version = -1
    last_flir_version = -1

    base_event_image = None
    base_rgb_image = None

    while not stop_event.is_set():
        try:
            if system_state.value != 't':
                time.sleep(0.01)
                continue

            current_event_version = event_data_version.value
            current_flir_version = flir_data_version.value
            if current_event_version == last_event_version and current_flir_version == last_flir_version:
                time.sleep(0.001)
                continue
            event_updated = current_event_version != last_event_version
            flir_updated = current_flir_version != last_flir_version
            last_event_version = current_event_version
            last_flir_version = current_flir_version

            # Update event visualization and current time when events update
            if event_updated or base_event_image is None:
                current_num_events = num_event.value
                if current_num_events > 0 and current_num_events <= event_shape[0]:
                    event_data = event_buffer[:current_num_events].copy()
                    base_event_image = np.full((rgb_height, rgb_width, 3), 255, dtype=np.uint8)
                    if len(event_data) > 0:
                        last_event_time = event_data['t'][-1] / 1000000.0
                        for event in event_data:
                            x, y, polarity, _ = int(event['x']), int(event['y']), int(event['p']), int(event['t'])
                            if 0 <= x < rgb_width and 0 <= y < rgb_height:
                                if polarity == 1:
                                    base_event_image[y, x] = [0, 0, 255]
                                else:
                                    base_event_image[y, x] = [255, 0, 0]
                else:
                    base_event_image = np.full((rgb_height, rgb_width, 3), 255, dtype=np.uint8)

            if flir_updated or base_rgb_image is None:
                base_rgb_image = flir_buffer.copy()

            if base_event_image is None or base_rgb_image is None:
                time.sleep(0.001)
                continue

            # Update detection indices based on current event time
            if last_event_time is not None:
                for name, data in results.items():
                    idx = indexes.get(name, -1)
                    while idx + 1 < len(data) and data[idx + 1][3] <= last_event_time:
                        idx += 1
                    indexes[name] = idx

            rgb_tracking_image = base_rgb_image.copy()
            event_tracking_image = base_event_image.copy()

            for name, data in results.items():
                idx = indexes.get(name, -1)
                if idx < 0 or idx >= len(data):
                    continue
                x, y, size = data[idx][0], data[idx][1], data[idx][2]
                if x <= 0 or y <= 0 or size <= 0:
                    continue
                color = color_map.get(name, (255, 255, 255))
                half_size = int(size / 2)
                x_int, y_int = int(x), int(y)
                p1 = (x_int - half_size, y_int - half_size)
                p2 = (x_int + half_size, y_int + half_size)
                cv2.rectangle(rgb_tracking_image, p1, p2, color, box_thickness)
                cv2.circle(rgb_tracking_image, (x_int, y_int), 3, color, -1)
                cv2.rectangle(event_tracking_image, p1, p2, color, box_thickness)
                cv2.circle(event_tracking_image, (x_int, y_int), 3, color, -1)

            # Resize for display
            scale = processing['display_scale']
            rgb_height_scaled = int(rgb_tracking_image.shape[0] * scale)
            rgb_width_scaled = int(rgb_tracking_image.shape[1] * scale)
            rgb_resized = cv2.resize(rgb_tracking_image, (rgb_width_scaled, rgb_height_scaled), interpolation=cv2.INTER_AREA)

            event_height_scaled = int(event_tracking_image.shape[0] * scale)
            event_width_scaled = int(event_tracking_image.shape[1] * scale)
            event_resized = cv2.resize(event_tracking_image, (event_width_scaled, event_height_scaled), interpolation=cv2.INTER_AREA)

            max_height = max(rgb_resized.shape[0], event_resized.shape[0])
            if rgb_resized.shape[0] < max_height:
                pad_height = max_height - rgb_resized.shape[0]
                rgb_resized = cv2.copyMakeBorder(rgb_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if event_resized.shape[0] < max_height:
                pad_height = max_height - event_resized.shape[0]
                event_resized = cv2.copyMakeBorder(event_resized, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            combined_image = np.hstack([rgb_resized, event_resized])
            cv2.imshow('Tracking Playback: RGB & Event Cameras', combined_image)
            cv2.waitKey(1)

        except Exception as e:
            print(f"GUI Comparison Playback: Error in display loop: {e}")
            sys.stdout.flush()
            time.sleep(0.01)

    print("GUI Comparison Playback: Cleaning up...")
    sys.stdout.flush()
    cv2.destroyAllWindows()
    try:
        event_shm.close()
        flir_shm.close()
    except Exception as e:
        print(f"GUI Comparison Playback: Error closing shared memory: {e}")
        sys.stdout.flush()

    print("GUI Comparison Playback: Exit")
    sys.stdout.flush()
