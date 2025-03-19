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
    mask = distances <= 0.3
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
    print(compute_distance_stats(np.array(impact_point_real), np.array(impact_point_predict)))


if __name__ == '__main__':
    with open('../data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    draw_3D_trajetory(loaded_data,0)
