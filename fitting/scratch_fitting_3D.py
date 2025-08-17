import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import rotate_matrix_coordinate
from fitting.parabola_fitter import MonoRansac
import matplotlib.pyplot as plt


def index_to_position(point_index,y_bias=0,z_bias=0.2):
    if point_index>0:
        row=point_index//10
        col=point_index%10
        y=(col)*0.1+y_bias+0.05
        z=(row)*0.1+z_bias+0.05
        return [0,y,z]
    else:
        point_index=-point_index
        row = point_index // 10
        col = point_index % 10
        # print(col,row)
        y = -(col) * 0.1 + y_bias + 0.05
        z = (row) * 0.1 + z_bias + 0.05
        return [0, y, z]

def position_to_index(position, y_bias=0, z_bias=0.2):
    """Convert a 3D position to a grid index."""
    y = position[1] - y_bias - 0.05
    z = position[2] - z_bias - 0.05
    row = int(z // 0.1)
    col = int(y // 0.1)
    print("row:", row, "col:", col, "y:", y, "z:", z)
    if y>=0:
        return -row * 10 + col
    else:
        return row * 10 - col


def plot_2d_impact_comparison(exp_results, success_threshold=0.25):
    """
    Plot 2D comparison of real vs predicted impact points with success/failure marking
    
    Args:
        exp_results: List of experiment results with 'impact' and 'label' keys
        success_threshold: Distance threshold for successful prediction (meters)
    """
    real_y = []
    real_z = []
    pred_y = []
    pred_z = []
    success_flags = []
    trajectory_ids = []
    
    for result in exp_results:
        real_impact = result['label']
        pred_impact = result['impact']
        traj_id = result.get('trajectory_id', 0)
        
        # Extract Y and Z coordinates (ignoring X=0)
        real_y.append(real_impact[1])
        real_z.append(real_impact[2])
        pred_y.append(pred_impact[1])
        pred_z.append(pred_impact[2])
        trajectory_ids.append(traj_id)
        
        # Calculate Euclidean distance for success determination
        distance = np.sqrt((real_impact[1] - pred_impact[1])**2 + 
                          (real_impact[2] - pred_impact[2])**2)
        success_flags.append(distance <= success_threshold)
    
    # Convert to numpy arrays
    real_y = np.array(real_y)
    real_z = np.array(real_z)
    pred_y = np.array(pred_y)
    pred_z = np.array(pred_z)
    success_flags = np.array(success_flags)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    
    # Plot successful predictions in green
    successful_mask = success_flags
    plt.scatter(real_y[successful_mask], real_z[successful_mask], 
               c='green', marker='o', s=100, alpha=0.7, label='Real (Success)')
    plt.scatter(pred_y[successful_mask], pred_z[successful_mask], 
               c='lightgreen', marker='x', s=100, alpha=0.7, label='Predicted (Success)')
    
    # Plot failed predictions in red
    failed_mask = ~success_flags
    plt.scatter(real_y[failed_mask], real_z[failed_mask], 
               c='red', marker='o', s=100, alpha=0.7, label='Real (Failed)')
    plt.scatter(pred_y[failed_mask], pred_z[failed_mask], 
               c='pink', marker='x', s=100, alpha=0.7, label='Predicted (Failed)')
    
    # Draw connection lines between real and predicted points
    for i in range(len(real_y)):
        color = 'green' if success_flags[i] else 'red'
        alpha = 0.3
        plt.plot([real_y[i], pred_y[i]], [real_z[i], pred_z[i]], 
                color=color, alpha=alpha, linewidth=1)
    
    # Add trajectory ID annotations for failed cases
    for i in range(len(real_y)):
        if not success_flags[i]:
            plt.annotate(f'{trajectory_ids[i]}', 
                        (real_y[i], real_z[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # Calculate statistics
    success_rate = np.sum(success_flags) / len(success_flags) * 100
    mean_error = np.mean([np.sqrt((real_y[i] - pred_y[i])**2 + (real_z[i] - pred_z[i])**2) 
                         for i in range(len(real_y))])
    
    plt.xlabel('Y Coordinate (m)')
    plt.ylabel('Z Coordinate (m)')
    plt.title(f'Impact Point Prediction Results\n'
              f'Success Rate: {success_rate:.1f}% | Mean Error: {mean_error:.3f}m | '
              f'Threshold: {success_threshold}m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set display range limits
    plt.xlim(-1, 1)
    plt.ylim(0, 2)
    
    # Add statistics text box
    stats_text = f'Total Trajectories: {len(exp_results)}\n' \
                f'Successful: {np.sum(success_flags)}\n' \
                f'Failed: {np.sum(~success_flags)}\n' \
                f'Success Rate: {success_rate:.1f}%\n' \
                f'Mean Error: {mean_error:.3f}m'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Prediction Results ===")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Error: {mean_error:.3f}m")
    print(f"Success Threshold: {success_threshold}m")
    print(f"Total Trajectories: {len(exp_results)}")
    print(f"Successful Predictions: {np.sum(success_flags)}")
    print(f"Failed Predictions: {np.sum(~success_flags)}")


def main():
    """ Main """

    plot_index=29
    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixel_3D_records_RGB_c"
    # camera_config = {
    #     'fx': 739.9,
    #     'fy': 739.9,
    #     'cx': 310,
    #     'cy': 710,
    #     'dist_'
    #     'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]
    # }


    camera_config = {
        "fx": 987.1678420424932,
        "fy": 986.0683237433913,
        "cy": 739.7418995490075,
        "cx": 513.7733221750243,
        "dist_coeffs": [
            -0.11123899870851763,
            -0.020663002330730863,
            -0.0002997445722290338,
            0.0007964973013426446,
            0.014818755754515196
        ]
    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])

    label_file= "/evaluation_tools/label_c.csv"
    labels=np.loadtxt(label_file, delimiter=",")
    print(labels)
    # fitter = MonoFitter3D(camera_params=camera_config,rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter(camera_params=camera_config, rotation_matrix_to=rotation_matrix_to,
    #                          rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter3DRansac(camera_params=camera_config, rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter(camera_params=camera_config, rotation_matrix_back=rotation_matrix_back)
    # fitter = PixelSizeFitter(camera_params=camera_config,rotation_matrix_back=rotation_matrix_back)
    fitter = MonoRansac(camera_params=camera_config,rotation_matrix_to=rotation_matrix_to,rotation_matrix_back=rotation_matrix_back)
    exp_results = []

    for i in range(0,len(labels)):
        # Skip invalid data where label is 0
        if labels[i] == 0:
            print(f"Skipping trajectory {i}: invalid label (0)")
            continue
            
        ball_memory=np.loadtxt(f"{save_dir}/{i}.csv", delimiter=",")
        print(ball_memory)
        print(i)
        observations=ball_memory
        observations[:,3]=observations[:,3]-observations[0,3]
        observations= observations[:20, :]
        fitter.set_data(observations)
        result=fitter.get_params()

        impact_point_real=index_to_position(labels[i])
        ball = result['points']

        impact_point=fitter.predict()

        dict_to_save={'points' : ball,
             'impact':impact_point,
             'label':impact_point_real,
             'fit_params': result.get('params', None),
             'inliers': result.get('inliers', None),
             'trajectory_id': i,  # Add original trajectory ID
             }
        exp_results.append(dict_to_save)
    
    # Comment out 3D visualization (function preserved but not called)
    # draw_multi_3D_trajetory_cls(raw_data=exp_results)
    
    # Add 2D visualization for impact points
    plot_2d_impact_comparison(exp_results)

if __name__ == "__main__":
    main()
    # print(cal_position(14))