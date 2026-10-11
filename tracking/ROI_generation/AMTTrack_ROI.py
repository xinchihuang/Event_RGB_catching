import cv2
import os
import sys
from multiprocessing import shared_memory
import numpy as np
import time
import torch

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config


# Automatic configuration detection is disabled to avoid interfering with runtime modes.
# def ensure_correct_config():
#     """Ensure the correct configuration file is loaded."""
#     current_bbox = Config.get_roi()['default_bbox']
#     # If the bbox matches the default configuration, try config_l.yaml.
#     if current_bbox == [430, 1180, 40, 40]:
#         try:
#             switch_global_config('config/config_l.yaml')
#             print("Auto-switched to config_l.yaml for development")
#         except:
#             pass

# # Automatic configuration detection, commented out to avoid interference.
# ensure_correct_config()
def transform_points_to_voxels(event_data, voxel_generator=None, device=torch.device("cuda")):
    """
    Convert the point cloud to voxels using spconv's voxel generator.
    """
    save_voxel = 5000
    x = event_data['x']  # Shape: (N,)
    # print(x)
    y = event_data['y']  # Shape: (N,)
    p = event_data['p']  # Shape: (N,)
    t = event_data['t']  # Shape: (N,)
    # print(t)

    time_length = t[-1] - t[0]
    # print(time_length)
    if time_length == 0:
        time_length = 1e-6
    t = ((t - t[0]) / time_length) * 1000

    current_events = np.stack((t, x, y, p), axis=-1)
    current_events = torch.tensor(current_events, device=device).float()
    # print(current_events)
    voxel_output = voxel_generator(current_events)

    # For an N*4 point cloud, pillar generation returns three arrays.
    # voxels has shape [M, max_num_points_per_voxel, 4] for (t, x, y, p).
    # coordinates contains zyx voxel coordinates, with shape [M, 3].
    # num_points has shape [M] and counts valid points before padding.
    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # Keep at most save_voxel voxels, ranked by event count.
    # print(torch.where(num_points>=16)[0].shape)
    if num_points.shape[0] < save_voxel:
        features = voxels[:, :, 3]
        coor = coordinates[:, :]
    else:
        _, voxels_idx = torch.topk(num_points, save_voxel)
        # Use each voxel's point polarities as its feature vector.
        features = voxels[voxels_idx][:, :, 3]
        # Coordinates of the selected voxels.
        coor = coordinates[voxels_idx]
    # Reorder y,x,t to t,x,y.
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features


def model(shared_event_data, shared_flir_data, shared_ROI_data, num_event, event_data_version, flir_data_version, system_state, stop_event):
    import sys
    import os
    # Add AMTTrack directory to path (repo root / AMTTrack)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    amttrack_path = os.path.join(repo_root, "AMTTrack")
    if amttrack_path not in sys.path:
        sys.path.append(amttrack_path)
    from lib.test.evaluation.tracker import Tracker
    from spconv.pytorch.utils import PointToVoxel

    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    # Load the correct configuration in the child process.
    # Check the environment to select the configuration.
    active_config = os.environ.get('ACTIVE_CONFIG_FILE')
    active_task = os.environ.get('ACTIVE_TASK')

    if active_config:
        print(f"Model process: Using config from environment: {active_config} (task: {active_task})")
        # Config will be automatically loaded based on environment variables
    else:
        print("Model process: No active config in environment, using default config")

    print(f"Model process using templates: {Config.get_paths()['template_npy']}, {Config.get_paths()['template_jpg']}")
    print(f"Model process using default_bbox: {Config.get_roi()['default_bbox']}")
    print(f"Model process using record_dir: {Config.get_paths()['record_dir']}")

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']
    event_width, event_height = dims['event_width'], dims['event_height']

    # Get image processing parameters from config
    processing = Config.get_image_processing()

    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=processing['fps'],
                                                       palette=ColorPalette.Dark)

    def detect_initial_bbox(image, event_data):

        """
        Automatically detect the initial ball position.
        """
        # Import the ball detector.
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        sys.path.append(current_dir)

        # Generate an event image for detection and overlay display.
        # Note: This is called once during initialization, so creating a local buffer is acceptable
        event_frame_local = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        event_frame_gen.generate_frame(event_data, event_frame_local,
                                       accumulation_time_us=processing['accumulation_time_us'])
        event_image_gray = cv2.cvtColor(event_frame_local, cv2.COLOR_BGR2GRAY)

        # Binarize the event image for visualization.
        _, binary = cv2.threshold(event_image_gray, processing['binary_threshold'], processing['binary_max_value'],
                                  cv2.THRESH_BINARY)
        event_image_viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Attempt automatic ball detection.
        # detected_bbox = ball_detector.detect_ball(image, event_image_gray)
        # print(detected_bbox)
        detected_bbox = Config.get_roi()['default_bbox']
        print(detected_bbox)
        if detected_bbox is not None:
            x, y, w, h = detected_bbox
            print(f"Automatically detected ball position: x={x}, y={y}, w={w}, h={h}")

            # Draw the detected bbox on the RGB image.
            vis_image = image.copy()
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, "Auto Detected Ball", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw the bbox on the event image too.
            cv2.rectangle(event_image_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(event_image_viz, "Event View", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Overlay the RGB and event images.
            combined = np.hstack((vis_image, event_image_viz))

            # The preview is shown by the parent process before child processes
            # start. OpenCV/Qt GUI calls from multiple processes can deadlock.

            return x, y, w, h
        else:
            # Use the default position if automatic detection fails.
            print("Automatic detection failed; using the default position")
            roi_config = Config.get_roi()
            if 'backup_bbox' in roi_config:
                x, y, w, h = roi_config['backup_bbox']
            else:
                # Use default_bbox if backup_bbox is unavailable.
                x, y, w, h = roi_config['default_bbox']
            return x, y, w, h

    device = torch.device("cuda")
    voxel_generator = PointToVoxel(
        # Voxel dimensions along normalized time, x, and y.
        vsize_xyz=[1000, 10, 10],
        # Lower and upper bounds for (t, x, y).
        coors_range_xyz=[0., 0, 0, 100000, event_width, event_height],
        # Each point contains (t, x, y, polarity).
        num_point_features=4,
        # Maximum number of generated voxels.
        max_num_voxels=16000,  # 16000
        # Maximum number of sampled points per voxel.
        max_num_points_per_voxel=16,  # 1024
        device=device
    )

    start_time = time.time()

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    first_flir_image = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

    # Resolve an absolute path for child processes (templates are relative to tracking/).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_path = os.path.join(project_root, Config.get_paths()['template_npy'])
    first_event_data = np.load(npy_path, allow_pickle=True)

    coor, features = transform_points_to_voxels(event_data=first_event_data, voxel_generator=voxel_generator,
                                                device=device)
    event_template = torch.cat((coor, features), dim=1)

    # event_template = torch.zeros([1024, 19])

    # Resolve an absolute path so child processes can find the file.
    jpg_path = os.path.join(project_root, Config.get_paths()['template_jpg'])
    first_flir_image = cv2.imread(jpg_path)

    # Apply brightness enhancement from config
    processing = Config.get_image_processing()
    brightness_multiplier = processing['brightness_multiplier']
    if brightness_multiplier != 1.0:
        first_flir_image = np.clip(first_flir_image.astype(np.float32) * brightness_multiplier, 0, 255).astype(np.uint8)

    initial_bbox = detect_initial_bbox(first_flir_image, first_event_data)
    # initial_bbox= (495, 1098, 40, 40)

    x, y, w, h = initial_bbox
    # cv2.rectangle(first_flir_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("a", first_flir_image)
    # cv2.waitKey(0)
    bbox = [x, y, w, h]
    init_info = {'init_bbox': bbox}

    count = 0
    roi_params = Config.get_roi()
    roi_range = roi_params['roi_range']
    initialized = False
    system_state.value = 'w'  # Initialize the state to waiting.

    # Pre-allocate event_frame buffer to avoid repeated allocation in loop
    event_frame_buffer = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)

    # Track last processed data versions to skip stale data
    last_event_version = -1
    last_flir_version = -1

    # Timing statistics
    from datetime import datetime
    process_start_time = time.time()
    frame_times = []

    while not stop_event.is_set():
        # Check if we have new data, skip if stale
        current_event_version = event_data_version.value
        if current_event_version == last_event_version:
            time.sleep(0.001)  # Brief sleep to avoid busy-waiting
            continue
        last_event_version = current_event_version

        event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()
        event_data = event_data[:num_event.value]

        if system_state.value == 'w' and not initialized:
            print('Model Initializing')
            # AMtracker = Tracker('amttrack', 'amttrack_felt', 'felt', None)
            AMtracker = Tracker('amttrack', 'amttrack_ball', 'ball', None)

            params = AMtracker.get_parameters()
            # print(params)
            params.debug = None
            tracker = AMtracker.create_tracker(params)

            out = tracker.initialize(first_flir_image, event_template, init_info, idx=0)
            # run one iter
            event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
            flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()

            start_time = time.time()
            x, y, w, h = initial_bbox
            bbox = [x, y, w, h]
            init_info = {'init_bbox': bbox}
            info = {'previous_output': {'target_bbox': bbox}, 'gt_bbox': bbox}
            voxel_start = time.time()
            coor, features = transform_points_to_voxels(event_data=event_data, voxel_generator=voxel_generator,
                                                        device=device)
            voxel_end = time.time()
            print("Voxel time:", voxel_end - voxel_start)
            # print(coor.shape, features.shape)
            event_search = torch.cat((coor, features), dim=1)
            tracker.state = bbox
            inference_start = time.time()
            out = tracker.track(flir_data, event_search, info)
            inference_end = time.time()
            print("Inference time:", inference_end - inference_start)
            # print("OUT",out)
            bbox = out.get('target_bbox')
            print("Model Ready")
            print("")
            initialized = True
            count = 0
            system_state.value = 't'

        if num_event.value == 0:
            continue
        if system_state.value == 't':
            # === Timing Section Start ===
            frame_start_absolute = datetime.now()
            frame_start_relative = time.time()
            frame_start_since_process = frame_start_relative - process_start_time

            info = {'previous_output': {'target_bbox': bbox}, 'gt_bbox': bbox}

            # Voxel transformation
            voxel_start = time.time()
            try:
                coor, features = transform_points_to_voxels(event_data=event_data, voxel_generator=voxel_generator,
                                                            device=device)
            except:
                continue
            voxel_end = time.time()
            voxel_time = voxel_end - voxel_start

            # Prepare event search tensor
            tensor_prep_start = time.time()
            event_search = torch.cat((coor, features), dim=1)
            if event_search == None:
                continue
            tracker.state = bbox
            tensor_prep_time = time.time() - tensor_prep_start

            # Neural network inference
            inference_start = time.time()
            out = tracker.track(flir_data, event_search, info)
            inference_end = time.time()
            inference_time = inference_end - inference_start

            # Post-processing and ROI update
            postprocess_start = time.time()
            bbox = out.get('target_bbox')
            confidence = out.get('confidence', 0.0)  # Get confidence, default to 0.0 if not available
            print('confidence', confidence)
            # Confidence-based tracking strategy
            if confidence < 0.15:  # Very low confidence - reset to image center
                center_x = rgb_width // 2
                center_y = rgb_height*2// 3
                center_size = 40
                bbox = [center_x - center_size // 2, center_y - center_size // 2, center_size, center_size]
                print(f"Frame {count}: Very low confidence {confidence:.3f}, reset to center: {bbox}")
                # Update info for next iteration
                info = {'previous_output': {'target_bbox': bbox}, 'gt_bbox': bbox}
            elif confidence < 0.2:  # Low confidence - keep current bbox but warn
                print(f"Frame {count}: Low confidence {confidence:.3f}, bbox: {bbox}")

            state = [int(s) for s in bbox]
            x_r, y_r, w_r, h_r = state

            ROI[0] = x_r
            ROI[1] = y_r
            ROI[2] = w_r
            ROI[3] = h_r
            postprocess_time = time.time() - postprocess_start
            # Visualization
            visualization_start = time.time()
            # Reuse pre-allocated buffer instead of creating new array
            event_frame = event_frame_buffer
            event_frame[:] = 0  # Clear buffer
            event_frame_gen.generate_frame(event_data, event_frame,
                                           accumulation_time_us=processing['accumulation_time_us'])
            image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(image, processing['binary_threshold'], processing['binary_max_value'],
                                      cv2.THRESH_BINARY)
            event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            rgb_image = flir_data
            # All drawing code removed - no visual feedback needed

            combined = np.hstack((rgb_image, event_image))
            scale = processing['display_scale']
            # Calculate the resized dimensions.
            width = int(combined.shape[1] * scale)
            height = int(combined.shape[0] * scale)
            new_size = (width, height)

            resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
            #
            # cv2.imshow('Image with Bounding Box', resized_img)
            # cv2.waitKey(1)
            # cv2.imwrite(f"{root_dir}/{0}/combined/" + str(count) + ".jpg", combined)
            visualization_time = time.time() - visualization_start

            # === Timing Section End ===
            frame_end_relative = time.time()
            frame_total_time = frame_end_relative - frame_start_relative
            frame_times.append(frame_total_time)

            # Print detailed timing every 10 frames (AMTTrack is slower, less frequent logging)
            if count % 10 == 0:
                avg_frame_time = np.mean(frame_times[-10:]) if len(frame_times) >= 10 else np.mean(frame_times)
                print(f"\n{'='*80}")
                print(f"AMTTRACK MODEL TIMING REPORT - Frame #{count}")
                print(f"{'='*80}")
                print(f"Absolute Time:        {frame_start_absolute.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"Time Since Start:     {frame_start_since_process:.3f}s")
                print(f"Confidence:           {confidence:.3f}")
                print(f"BBox Output:          [{x_r}, {y_r}, {w_r}, {h_r}]")
                print(f"{'-'*80}")
                print(f"Phase Breakdown:")
                print(f"  Voxel Transform:    {voxel_time*1000:6.2f}ms ({voxel_time/frame_total_time*100:5.1f}%)")
                print(f"  Tensor Prep:        {tensor_prep_time*1000:6.2f}ms ({tensor_prep_time/frame_total_time*100:5.1f}%)")
                print(f"  NN Inference:       {inference_time*1000:6.2f}ms ({inference_time/frame_total_time*100:5.1f}%) *** BOTTLENECK ***")
                print(f"  Post-processing:    {postprocess_time*1000:6.2f}ms ({postprocess_time/frame_total_time*100:5.1f}%)")
                print(f"  Visualization:      {visualization_time*1000:6.2f}ms ({visualization_time/frame_total_time*100:5.1f}%)")
                print(f"{'-'*80}")
                print(f"Total Frame Time:     {frame_total_time*1000:6.2f}ms")
                print(f"Avg Frame Time (10):  {avg_frame_time*1000:6.2f}ms")
                print(f"Estimated FPS:        {1.0/avg_frame_time:6.2f}")
                print(f"{'='*80}\n")

            count += 1
            initialized = False
