import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt


def index_to_position(point_index, y_bias=-0.1, z_bias=0):
    """Convert grid index to 3D position coordinates"""
    if point_index > 0:
        row = point_index // 10
        col = point_index % 10
        y = (col) * 0.1 + y_bias + 0.05
        z = (row) * 0.1 + z_bias + 0.05
        return [0, y, z]
    else:
        point_index = -point_index
        row = point_index // 10
        col = point_index % 10
        y = -(col) * 0.1 + y_bias + 0.05
        z = (row) * 0.1 + z_bias + 0.05
        return [0, y, z]


def read_prediction_csv(csv_path):
    """
    Read landing prediction from CSV file
    Expected format: single line with x, y, z coordinates
    """
    try:
        data = np.loadtxt(csv_path, delimiter=",")
        if data.size == 0:
            print(f"Warning: Empty CSV file {csv_path}")
            return None
        
        if data.ndim == 1 and len(data) == 3:
            # Single row with x, y, z
            return data
        elif data.ndim == 2 and data.shape[1] == 3:
            # Multiple rows, take the first one
            print(f"Warning: Multiple rows in {csv_path}, using first row")
            return data[0]
        else:
            print(f"Error: Invalid data format in {csv_path}. Expected 3 columns (x,y,z)")
            return None
            
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def plot_2d_prediction_comparison(exp_results, success_threshold=0.25):
    """
    Plot 2D comparison of real vs predicted landing points
    
    Args:
        exp_results: List of experiment results with 'prediction' and 'label' keys
        success_threshold: Distance threshold for successful prediction (meters)
    """
    real_y = []
    real_z = []
    pred_y = []
    pred_z = []
    success_flags = []
    trajectory_ids = []
    
    for result in exp_results:
        real_landing = result['label']
        pred_landing = result['prediction']
        traj_id = result['trajectory_id']
        
        # Extract Y and Z coordinates (ignoring X=0)
        real_y.append(real_landing[1])
        real_z.append(real_landing[2])
        pred_y.append(pred_landing[1])
        pred_z.append(pred_landing[2])
        trajectory_ids.append(traj_id)
        
        # Calculate Euclidean distance for success determination
        distance = np.sqrt((real_landing[1] - pred_landing[1])**2 + 
                          (real_landing[2] - pred_landing[2])**2)
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
    success_rate = np.sum(success_flags) / len(success_flags) * 100 if len(success_flags) > 0 else 0
    mean_error = np.mean([np.sqrt((real_y[i] - pred_y[i])**2 + (real_z[i] - pred_z[i])**2) 
                         for i in range(len(real_y))]) if len(real_y) > 0 else 0
    
    plt.xlabel('Y Coordinate (m)')
    plt.ylabel('Z Coordinate (m)')
    plt.title(f'Landing Point Prediction Results\\n'
              f'Success Rate: {success_rate:.1f}% | Mean Error: {mean_error:.3f}m | '
              f'Threshold: {success_threshold}m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set display range limits
    plt.xlim(-1, 1)
    plt.ylim(0, 2)
    
    # Add statistics text box
    stats_text = f'Total Trajectories: {len(exp_results)}\\n' \
                f'Successful: {np.sum(success_flags)}\\n' \
                f'Failed: {np.sum(~success_flags)}\\n' \
                f'Success Rate: {success_rate:.1f}%\\n' \
                f'Mean Error: {mean_error:.3f}m'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.savefig("landing_point_comparison.png", dpi=300)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\n=== Landing Point Prediction Results ===")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Error: {mean_error:.3f}m")
    print(f"Success Threshold: {success_threshold}m")
    print(f"Total Trajectories: {len(exp_results)}")
    print(f"Successful Predictions: {np.sum(success_flags)}")
    print(f"Failed Predictions: {np.sum(~success_flags)}")


def main():
    """Main function to read CSV predictions and plot results"""
    
    # Configuration/
    save_dir = "/home/xinchi/Event_RGB_catching_data/RGB_normal/ball_pixel_3D_records"
    label_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_tools/label_l.csv")
    
    # Check if directories exist
    if not os.path.exists(save_dir):
        print(f"Error: Prediction directory {save_dir} does not exist")
        return
    
    if not os.path.exists(label_file):
        print(f"Error: Label file {label_file} does not exist")
        return
    
    # Load labels (ground truth)
    try:
        labels = np.loadtxt(label_file, delimiter=",")
        print(f"Loaded {len(labels)} labels from {label_file}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return
    
    exp_results = []
    
    # Read prediction files
    for i in range(len(labels)):
        # Skip invalid data where label is 0
        if labels[i] == 0:
            print(f"Skipping trajectory {i}: invalid label (0)")
            continue
        
        csv_path = os.path.join(save_dir, f"{i}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Skipping trajectory {i}: CSV file {csv_path} not found")
            continue
        
        # Read prediction from CSV
        prediction = read_prediction_csv(csv_path)
        if prediction is None:
            continue
        
        # Get real landing point from label
        real_landing = index_to_position(labels[i])
        
        # Store results
        result = {
            'prediction': prediction,
            'label': real_landing,
            'trajectory_id': i
        }
        exp_results.append(result)
        
        print(f"Trajectory {i}: Real={real_landing}, Predicted={prediction}")
    
    if not exp_results:
        print("No valid results found. Check your CSV files and labels.")
        return
    
    print(f"\\nProcessed {len(exp_results)} trajectories")
    
    # Plot comparison
    plot_2d_prediction_comparison(exp_results)


if __name__ == "__main__":
    main()