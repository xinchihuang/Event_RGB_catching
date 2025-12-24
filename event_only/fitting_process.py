"""
Ball trajectory fitting and robot control module for event-based ball catching system.
Implements trajectory prediction using parabolic motion models and motor control.
"""

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

def landing_prediction(shared_ball_pixel_3D_data, ball_observations, shared_impact_point_data, record_index, system_state, stop_event):
    """
    Predict ball landing position using trajectory fitting.
    
    Args:
        shared_ball_pixel_3D_data: Shared memory containing ball observation data
        ball_observations: Number of ball observations
        shared_impact_point_data: Shared memory for predicted impact point
        record_index: Current record index
        system_state: Current system state
        stop_event: Event to signal process termination
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from .parabola_fitter import MonoRansac
    from .utils import rotate_matrix_coordinate

    # Load camera intrinsics from configuration
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

    # Initialize shared memory arrays
    ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape = shared_ball_pixel_3D_data
    ball_pixel_shm = shared_memory.SharedMemory(name=ball_pixel_memory.name)
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
    
    # Initialize trajectory fitter
    fitter = MonoRansac(camera_params=camera_config, rotation_matrix_to=rotation_matrix_to, rotation_matrix_back=rotation_matrix_back)
    buffered_impact_point = [0, 0, 0]
    last_index = 0
    # Main trajectory prediction loop
    while not stop_event.is_set():
        ball_pixels = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_shm.buf)
        ball_pixels = ball_pixels[:ball_observations.value]
        
        try:
            if system_state.value == 't':
                if ball_observations.value == last_index:
                    continue
                else:
                    last_index = ball_observations.value

                # Get trajectory parameters from configuration
                trajectory = Config.get_trajectory()
                
                if len(ball_pixels) == trajectory['required_observations']:
                    # Fit trajectory and predict landing position
                    observations = ball_pixels
                    fitter.set_data(observations)
                    fitter.get_params()
                    impact_point = fitter.predict()
                    bounds_x = trajectory['impact_bounds_x']
                    bounds_y = trajectory['impact_bounds_y']
                    
                    # Display prediction results
                    print(f"Predicted Landing Position: X={impact_point[0]:.3f}, Y={impact_point[1]:.3f}, Z={impact_point[2]:.3f}")
                    
                    # Validate prediction against bounds
                    if bounds_x[0] < impact_point[1] < bounds_x[1] and bounds_y[0] < impact_point[2] < bounds_y[1]:
                        buffered_impact_point = impact_point
                        print(f"Landing position within bounds - Using prediction")
                    else:
                        impact_point = buffered_impact_point
                        print(f"Landing position out of bounds - Using buffered position: X={impact_point[0]:.3f}, Y={impact_point[1]:.3f}, Z={impact_point[2]:.3f}")

                    # Update shared impact point for robot control
                    impact_point_array[0] = impact_point[0]
                    impact_point_array[1] = impact_point[1]
                    impact_point_array[2] = impact_point[2]
                    
                    # Display final results
                    print(f"Final Landing Position: X={impact_point_array[0]:.3f}, Y={impact_point_array[1]:.3f}, Z={impact_point_array[2]:.3f}")
                    print(f"Record Index: {record_index.value}, Ball Observations: {ball_observations.value}")
                    print("-" * 50)

        except Exception as e:
            print(f"Error in trajectory prediction: {e}")
            continue
