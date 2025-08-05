import plotly.graph_objects as go
import numpy as np
import pickle
import plotly.graph_objects as go

def compute_distance_stats(array1, array2):
    """
       计算 array1[i] 和 array2[i] 之间的欧式距离，并返回这些距离的均值和方差。
       假设 array1 和 array2 的形状完全一致，比如 (M, D)。
       """
    # 1. 如果形状不同，需要先确保它们能一一对应
    assert array1.shape == array2.shape, "array1 和 array2 的形状必须相同"

    # 2. 计算每一行对应点的距离
    #    (array1 - array2) 会得到形状 (M, D)
    #    np.linalg.norm(..., axis=1) 会对第2维进行范数计算，得到形状 (M,)
    distances = np.linalg.norm(array1 - array2, axis=1)
    mask = distances <= 0.9
    print(len(distances))
    filtterd_distances=distances[mask]
    print(len(filtterd_distances))
    # 3. 计算这些距离的均值和方差
    mean = np.mean(filtterd_distances)
    variance = np.var(filtterd_distances)
    return mean, variance
def draw_3D_trajetory(data, i=0):
    """
    Visualizes the 3D trajectory of a moving ball and compares predicted and actual impact points.

    Parameters:
    - data (dict): A dictionary containing:
        - 'impact' (tuple): The predicted impact point (impact_x, impact_y, impact_z).
        - 'label' (tuple): The actual impact point (label_x, label_y, label_z).
        - 'points' (ndarray): An array of shape (N, 3) representing the observed trajectory (x, y, z).
    - i (int): An index identifier for distinguishing multiple trajectories.

    Returns:
    - None (Displays a 3D Plotly figure)
    """

    fig = go.Figure()  # Create a new 3D figure

    # Extract impact and label (ground truth) coordinates
    impact_x, impact_y, impact_z = data['impact']
    label_x, label_y, label_z = data['label']

    # Extract trajectory points
    ball = data['points']
    x = ball[:, 0]  # X-coordinates of the trajectory
    y = ball[:, 1]  # Y-coordinates of the trajectory
    z = ball[:, 2]  # Z-coordinates of the trajectory

    # Plot the predicted impact point (Red)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[impact_y],
        z=[impact_z],
        mode='markers',
        name=f'predicted {i}',
        marker=dict(
            size=8,
            color='red',  # Red marker for predicted impact
            symbol='circle'
        ),
        showlegend=True
    ))

    # Plot the actual impact point (Green)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[label_y],
        z=[label_z],
        mode='markers',
        name=f'real {i}',
        marker=dict(
            size=8,
            color='green',  # Green marker for actual impact
            symbol='circle'
        ),
        showlegend=True
    ))

    # Plot the observed trajectory of the ball
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        name=f'observation {i}',
        line=dict(width=2)
    ))

    # Configure the 3D scene settings
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title='X', range=[0, 4], dtick=1),
    #         yaxis=dict(title='Y', range=[-1, 1], dtick=1),
    #         zaxis=dict(title='Z', range=[0, 3], dtick=1),
    #         camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))  # Set the camera perspective
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=30)  # Adjust figure margins
    # )

    # Plot the camera position (Yellow)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        name=f'camera',
        marker=dict(
            size=20,
            color='yellow',  # Yellow marker representing the camera position
            symbol='circle'
        ),
        showlegend=True
    ))

    # Show the 3D plot
    fig.show()
def draw_multi_3D_trajetory(raw_data):
    """
    Visualizes the 3D trajectory of a moving ball and compares predicted and actual impact points.

    Parameters:
    - data (dict): A dictionary containing:
        - 'impact' (tuple): The predicted impact point (impact_x, impact_y, impact_z).
        - 'label' (tuple): The actual impact point (label_x, label_y, label_z).
        - 'points' (ndarray): An array of shape (N, 3) representing the observed trajectory (x, y, z).
    - i (int): An index identifier for distinguishing multiple trajectories.

    Returns:
    - None (Displays a 3D Plotly figure)
    """
    impact_point_real = []
    impact_point_predict = []
    fig = go.Figure()  # Create a new 3D figure
    for i in range(len(raw_data)):
        data=raw_data[i]
        # Extract impact and label (ground truth) coordinates
        impact_x, impact_y, impact_z = data['impact']
        label_x, label_y, label_z = data['label']

        # Extract trajectory points
        ball = data['points']
        x = ball[:, 0]  # X-coordinates of the trajectory
        y = ball[:, 1]  # Y-coordinates of the trajectory
        z = ball[:, 2]  # Z-coordinates of the trajectory
        impact_point_predict.append([impact_y, impact_z])
        impact_point_real.append([label_y, label_z])

        # Plot the predicted impact point (Red)
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[impact_y],
            z=[impact_z],
            mode='markers',
            name=f'predicted {i}',
            marker=dict(
                size=8,
                color='red',  # Red marker for predicted impact
                symbol='circle'
            ),
            showlegend=True
        ))

        # Plot the actual impact point (Green)
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[label_y],
            z=[label_z],
            mode='markers',
            name=f'real {i}',
            marker=dict(
                size=8,
                color='green',  # Green marker for actual impact
                symbol='circle'
            ),
            showlegend=True
        ))

        # Plot the observed trajectory of the ball
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'observation {i}',
            line=dict(width=2)
        ))

    # # Configure the 3D scene settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, 4], dtick=1),
            yaxis=dict(title='Y', range=[-1, 1], dtick=1),
            zaxis=dict(title='Z', range=[0, 3], dtick=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))  # Set the camera perspective
        ),
        margin=dict(l=0, r=0, b=0, t=30)  # Adjust figure margins
    )

    # Plot the camera position (Yellow)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        name=f'camera',
        marker=dict(
            size=20,
            color='yellow',  # Yellow marker representing the camera position
            symbol='circle'
        ),
        showlegend=True
    ))

    # Show the 3D plot
    fig.show()
    print(compute_distance_stats(np.array(impact_point_real), np.array(impact_point_predict)))



def draw_multi_3D_trajetory_cls(raw_data):
    """
    Visualizes the 3D trajectory of a moving ball and compares predicted and actual impact points.
    Categories the distance between predicted and actual landing points and colors them accordingly.

    Parameters:
    - raw_data (list): A list of dictionaries containing:
        - 'impact' (tuple): The predicted impact point (impact_x, impact_y, impact_z).
        - 'label' (tuple): The actual impact point (label_x, label_y, label_z).
        - 'points' (ndarray): An array of shape (N, 3) representing the observed trajectory (x, y, z).

    Returns:
    - None (Displays a 3D Plotly figure)
    """

    def compute_distance(point1, point2):
        """
        计算两点之间的欧几里得距离。
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    def add_yz_grid(fig, y_range=(-1, 1), z_range=(0, 3), grid_size=0.1, x_pos=0):
        """
        在y-z平面上添加0.1x0.1的方格网格
        """
        # 生成网格线的坐标点
        y_lines = np.arange(y_range[0], y_range[1] + grid_size, grid_size)
        z_lines = np.arange(z_range[0], z_range[1] + grid_size, grid_size)
        
        # 添加垂直线（沿z方向）
        for y in y_lines:
            fig.add_trace(go.Scatter3d(
                x=[x_pos, x_pos],
                y=[y, y],
                z=[z_range[0], z_range[1]],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 添加水平线（沿y方向）
        for z in z_lines:
            fig.add_trace(go.Scatter3d(
                x=[x_pos, x_pos],
                y=[y_range[0], y_range[1]],
                z=[z, z],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

    impact_point_real = []
    impact_point_predict = []
    fig = go.Figure()  # Create a new 3D figure
    
    # 添加y-z平面的网格
    add_yz_grid(fig)
    
    success_count = 0
    for i in range(len(raw_data)):
        data = raw_data[i]
        # Extract impact and label (ground truth) coordinates
        impact_x, impact_y, impact_z = data['impact']
        label_x, label_y, label_z = data['label']

        # Extract trajectory points
        ball = data['points']
        x = ball[:, 0]  # X-coordinates of the trajectory
        y = ball[:, 1]  # Y-coordinates of the trajectory
        z = ball[:, 2]  # Z-coordinates of the trajectory
        impact_point_predict.append([impact_y, impact_z])
        impact_point_real.append([label_y, label_z])

        # Calculate the distance between predicted and actual landing points
        distance = compute_distance([impact_x, impact_y, impact_z], [label_x, label_y, label_z])

        # Set color based on the distance
        color = 'blue' if distance <= 0.25 else 'red'  # Blue for small distance, Red for large distance
        name = 'Success' if distance <= 0.25 else 'Fail'
        success_count+=1 if distance <= 0.25 else 0
        
        # Get original trajectory ID if available
        traj_id = data.get('trajectory_id', i)
        
        # Plot the predicted impact point
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[impact_y],
            z=[impact_z],
            mode='markers',
            name=f'Traj {traj_id}: Predicted ({name})',
            marker=dict(
                size=8,
                color=color,  # Color based on distance
                symbol='circle'
            ),
            showlegend=True
        ))

        # Plot the actual impact point (real)
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[label_y],
            z=[label_z],
            mode='markers',
            name=f'Traj {traj_id}: Real',
            marker=dict(
                size=8,
                color='green',  # Green for real impact points
                symbol='square'  # Different symbol to distinguish from predicted
            ),
            showlegend=True
        ))

        # Plot connection line between predicted and real impact points
        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[impact_y, label_y],
            z=[impact_z, label_z],
            mode='lines',
            name=f'{i}: Error Line',
            line=dict(
                color='gray',
                width=2,
                dash='dash'  # Dashed line to show connection
            ),
            showlegend=False,  # Don't clutter legend with error lines
            hovertemplate=f'<b>Trajectory {i}</b><br>Distance: {distance:.3f}m<extra></extra>'
        ))

        # Plot the observed trajectory of the ball with RANSAC inlier/outlier distinction
        if 'inliers' in data and data['inliers'] is not None:
            inliers = data['inliers']
            # Plot inliers (points used for fitting)
            inlier_indices = np.where(inliers)[0]
            outlier_indices = np.where(~inliers)[0]
            
            if len(inlier_indices) > 0:
                fig.add_trace(go.Scatter3d(
                    x=x[inlier_indices], y=y[inlier_indices], z=z[inlier_indices],
                    mode='markers',
                    name=f'inliers {i} ({len(inlier_indices)} pts)',
                    marker=dict(size=8, color='darkgreen', symbol='circle'),
                    showlegend=True
                ))
            
            if len(outlier_indices) > 0:
                fig.add_trace(go.Scatter3d(
                    x=x[outlier_indices], y=y[outlier_indices], z=z[outlier_indices],
                    mode='markers',
                    name=f'outliers {i} ({len(outlier_indices)} pts)',
                    marker=dict(size=6, color='red', symbol='x'),
                    showlegend=True
                ))
            
            # Connect all points with a thin line
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f'trajectory {i}',
                line=dict(width=1, color='gray'),
                showlegend=False
            ))
        else:
            # Fallback to original plotting if no inlier info
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=f'observation {i}',
                line=dict(width=2)
            ))

        # Plot the fitted parabolic trajectory if fit_params are available
        if 'fit_params' in data and data['fit_params'] is not None:
            fit_params = data['fit_params']
            x0, y0, z0 = fit_params['x0'], fit_params['y0'], fit_params['z0']
            vx, vy, vz = fit_params['vx'], fit_params['vy'], fit_params['vz']
            
            # Generate fitted trajectory points
            t_start = 0
            t_end = abs(x0 / vx) if vx != 0 else 1.0  # Time to reach x=0
            t_fit = np.linspace(t_start, t_end, 50)
            
            x_fit = x0 + vx * t_fit
            y_fit = y0 + vy * t_fit
            z_fit = z0 + vz * t_fit - 0.5 * 9.81 * t_fit ** 2
            
            fig.add_trace(go.Scatter3d(
                x=x_fit, y=y_fit, z=z_fit,
                mode='lines',
                name=f'fitted trajectory {i}',
                line=dict(width=3, dash='dot', color='orange'),
                showlegend=True
            ))

    # Configure the 3D scene settings with equal aspect ratio
    # Calculate aspect ratios based on data ranges to ensure equal scaling
    x_range = 4 - 0  # 4
    y_range = 1 - (-1)  # 2
    z_range = 3 - 0  # 3
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, 4], dtick=1),
            yaxis=dict(title='Y', range=[-1, 1], dtick=1),
            zaxis=dict(title='Z', range=[0, 3], dtick=1),
            aspectmode='manual',  # Manual aspect ratio control
            aspectratio=dict(x=x_range, y=y_range, z=z_range),  # Proportional to data ranges
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))  # Set the camera perspective
        ),
        margin=dict(l=0, r=0, b=0, t=30)  # Adjust figure margins
    )

    # Plot the camera position (Yellow)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        name=f'camera',
        marker=dict(
            size=20,
            color='yellow',  # Yellow marker representing the camera position
            symbol='circle'
        ),
        showlegend=True
    ))

    # Show the 3D plot
    fig.show()
    print(f"Success Rate: {success_count}/{len(raw_data)}")
    print(compute_distance_stats(np.array(impact_point_real), np.array(impact_point_predict)))



if __name__ == '__main__':
    with open('../data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    draw_3D_trajetory(loaded_data,0)
