from multiprocessing import shared_memory
import numpy as np
import time
import os
import sys

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def landing_prediction(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,record_index,system_state,stop_event,test_mode="Event",task="Normal"):

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from parabola_fitter import MonoRansacDrag
    from utils import rotate_matrix_coordinate
    import csv

    # camera_config = {
    #     'fx': 739.9,
    #     'fy': 739.9,
    #     'cx': 310,
    #     'cy': 710,
    #     'dist_'
    #     'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]
    #
    # }
    # Get camera intrinsics from config
    from config.config_loader import get_config
    config_loader = get_config()
    intrinsics = config_loader.get('camera_intrinsics', {})
    
    camera_config = {
        "fx": intrinsics.get('fx', 987.1678420424932),
        "fy": intrinsics.get('fy', 986.0683237433913),
        "cy": intrinsics.get('cy', 739.7418995490075),
        "cx": intrinsics.get('cx', 513.7733221750243),
        "dist_coeffs": intrinsics.get('distortion_coeffs', [
            -0.11123899870851763,
            -0.020663002330730863,
            -0.0002997445722290338,
            0.0007964973013426446,
            0.014818755754515196
        ])
    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])

    ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape = shared_ball_pixel_3D_data
    ball_pixel_shm = shared_memory.SharedMemory(name=ball_pixel_memory.name)
    # Create numpy array view ONCE outside the loop
    ball_pixels_full = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_shm.buf)

    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
    fitter = MonoRansacDrag(camera_params=camera_config,rotation_matrix_to=rotation_matrix_to,rotation_matrix_back=rotation_matrix_back)
    buffered_impact_point = [0,0,0]

    # Setup for saving ball_pixel_3D data based on test_mode and task
    # base_dir = f"/media/xinchi/data/evaluation_data/evaluation_results/{test_mode}_{task}"
    base_dir = f"/home/xinchi/real_test_data/test_results/{test_mode}_{task}"
    ball_pixel_save_dir = f"{base_dir}/ball_pixel_3D_records_ball"
    if not os.path.exists(ball_pixel_save_dir):
        os.makedirs(ball_pixel_save_dir)
    impact_pixel_save_dir = f"{base_dir}/ball_pixel_3D_records"
    if not os.path.exists(impact_pixel_save_dir):
        os.makedirs(impact_pixel_save_dir)
    ball_pixels_data_list = []  # Store ball_pixels data for current record
    saved=True
    pending_ball_pixels = None
    pending_impact_point = None
    pending_record_index = None
    first_ball_ts = None
    last_nonempty_ball_pixels = None
    save_delay_sec = 0.3
    try:
        while not stop_event.is_set():
            # Safely read ball_observations value and bound check
            current_obs = ball_observations.value
            if current_obs < 0 or current_obs > ball_pixel_shape[0]:
                continue
            ball_pixels = ball_pixels_full[:current_obs].copy()  # Make a copy to avoid race conditions
            # print(ball_pixels)
            # print("record_index", record_index.value,system_state.value)
            try:
                if system_state.value != 't':
                    first_ball_ts = None
                    last_nonempty_ball_pixels = None

                if system_state.value == 't':
                    if current_obs > 0:
                        last_nonempty_ball_pixels = ball_pixels
                        if first_ball_ts is None:
                            first_ball_ts = time.perf_counter()
                            saved = False

                    if first_ball_ts is None:
                        continue

                    if time.perf_counter() - first_ball_ts < save_delay_sec:
                        continue

                    snapshot = last_nonempty_ball_pixels if last_nonempty_ball_pixels is not None else ball_pixels

                    # Get trajectory parameters from config
                    trajectory = Config.get_trajectory()

                    # if len(snapshot) >= trajectory['required_observations']:
                    #     observations = snapshot
                    #     fitter.set_data(observations)
                    #     fitter.get_params()
                    #     impact_point = fitter.predict()
                    #     bounds_x = trajectory['impact_bounds_x']
                    #     bounds_y = trajectory['impact_bounds_y']
                    #     print(bounds_x,bounds_y,impact_point)
                    #     if bounds_x[0] < impact_point[1] < bounds_x[1] and bounds_y[0] < impact_point[2] < bounds_y[1]:
                    #         buffered_impact_point = impact_point
                    #     else:
                    #         impact_point = buffered_impact_point
                    # else:
                    #     impact_point = buffered_impact_point

                    observations = snapshot
                    fitter.set_data(observations)
                    fitter.get_params()
                    impact_point = fitter.predict()
                    bounds_x = trajectory['impact_bounds_x']
                    bounds_y = trajectory['impact_bounds_y']
                    print(bounds_x, bounds_y, impact_point)
                    if bounds_x[0] < impact_point[1] < bounds_x[1] and bounds_y[0] < impact_point[2] < bounds_y[1]:
                        buffered_impact_point = impact_point
                    else:
                        impact_point = buffered_impact_point

                    impact_point_array[0] = impact_point[0]
                    impact_point_array[1] = impact_point[1]
                    impact_point_array[2] = impact_point[2]

                    # Snapshot the data now so saving isn't affected by resets.
                    pending_ball_pixels = snapshot.copy()
                    pending_impact_point = impact_point_array.copy()
                    pending_record_index = record_index.value
                    print(impact_point_array)
                    # print("=== STATE CHANGE: 't' -> 'r' (Timer reached, ready to save) ===")
                    # system_state.value='r'
                    print("=== STATE CHANGE: 't' -> 'c' (Timer reached, ready to catch) ===")
                    system_state.value = 'c'

                if system_state.value == 'r' and not saved:
                    if pending_ball_pixels is None or pending_impact_point is None:
                        continue
                    record_idx = pending_record_index if pending_record_index is not None else record_index.value
                    impact = pending_impact_point.reshape((1, -1))

                    print("record_index", record_idx)

                    # Save observations to file using dynamic paths
                    filename = f"{impact_pixel_save_dir}/{record_idx}.csv"
                    np.savetxt(filename, impact, delimiter=',', fmt='%.6f')
                    print(f"Saved impact observations to {filename}")
                    filename_ball = f"{ball_pixel_save_dir}/{record_idx}.csv"
                    np.savetxt(filename_ball, pending_ball_pixels, delimiter=',', fmt='%.6f')
                    print(f"Saved ball observations to {filename_ball}")

                    saved= True
                    pending_ball_pixels = None
                    pending_impact_point = None
                    pending_record_index = None

            except:
                continue
    finally:
        print("Landing prediction: Cleaning up...")
        # Do NOT close shared memory in subprocess - main process handles cleanup
