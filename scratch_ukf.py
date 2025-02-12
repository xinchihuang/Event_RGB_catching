import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import plotly.graph_objects as go

# 参数设置
g = 9.8  # 重力加速度 (m/s^2)
cy = 640
cx = 360
fx = 673.797
fy = 673.797
dt = 0.1  # 时间间隔 (s)
num_steps = 100  # 时间步数

# 初始状态 [X, Y, Z, Vx, Vy, Vz]
initial_state = np.array([0, 0, 1, 10, 0, 0])
def rotate_matrix_coordinate(r_x,r_y,r_z):
    """
    Right hand coordinate rotation matrix for coordinate rotation
    Note this is for coordinate rotation only

    Args:
        r_x: rotation angle(degree) in x axis(counter-clockwise direction from zero) y->z
        r_y: rotation angle in y axis x->z
        r_z: rotation angle in z axis x->y

    Returns: rotation matrix in x-y-z order relative to the rotation axis

    """
    r_x=np.radians(-r_x)
    r_y=np.radians(-r_y)
    r_z=np.radians(-r_z)
    Rotate_x=np.array([
        [1, 0, 0],
        [0, np.cos(r_x), -np.sin(r_x)],
        [0, np.sin(r_x), np.cos(r_x)]
    ])
    Rotate_y=np.array([
        [np.cos(r_y), 0, np.sin(r_y)],
        [0, 1, 0],
        [-np.sin(r_y), 0, np.cos(r_y)]
    ])
    Rotate_z=np.array([
        [np.cos(r_z), -np.sin(r_z), 0],
        [np.sin(r_z), np.cos(r_z), 0],
        [0, 0, 1]
    ])
    return Rotate_z@Rotate_y@Rotate_x

# 生成真实轨迹
def generate_true_trajectory(initial_state, dt, num_steps):
    t = np.arange(0, num_steps * dt, dt)
    X = initial_state[0] + initial_state[3] * t
    Y = initial_state[1] + initial_state[4] * t
    Z = initial_state[2] + initial_state[5] * t - 0.5 * g * t**2
    return np.vstack([X, Y, Z]).T

# 生成带噪声的观测数据
def generate_noisy_observations(true_trajectory, R, t, H, noise_std):
    observations = []
    for point in true_trajectory:
        # 将世界坐标系中的点投影到图像平面
        u, v = H(np.append(point, [0, 0, 0]), R, t)

        # 添加噪声
        noise = np.random.normal(0, noise_std, 2)
        observations.append([u + noise[0], v + noise[1]])
    return np.array(observations)


# 状态转移函数（考虑重力的抛物线运动模型）
def F(x, dt):
    X, Y, Z, Vx, Vy, Vz = x
    X_new = X + Vx * dt
    Y_new = Y + Vy * dt
    Z_new = Z + Vz * dt - 0.5 * g * dt**2
    Vx_new = Vx
    Vy_new = Vy
    Vz_new = Vz - g * dt
    return np.array([X_new, Y_new, Z_new, Vx_new, Vy_new, Vz_new])



def H(x,R,t):
    """
    观测函数：将世界坐标系中的点投影到图像平面
    :param x: 状态向量 [X, Y, Z, Vx, Vy, Vz]
    :param R: 旋转矩阵 (3x3)
    :param t: 平移向量 (3x1)
    :return: 图像平面坐标 [u, v]
    """
    # 提取状态中的位置
    x_w, y_w, z_w = x[:3]

    # 将点从世界坐标系转换到摄像机坐标系
    point_world = np.array([x_w, y_w, z_w])
    point_camera = R @ point_world + t

    # 将摄像机坐标系中的点投影到图像平面
    x_c, y_c, z_c = point_camera
    u = fx * x_c / z_c + cx
    v = fy * y_c / z_c + cy

    return np.array([u, v])
# 初始化 UKF
def initialize_ukf(initial_state, dt,R,t):
    points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=2, dt=dt, fx=F, hx=lambda x:H(x,R,t), points=points)
    ukf.x = [0,0,0,00,0,0] # 初始状态
    ukf.P = np.diag([100, 100, 100, 10, 10, 10])
    ukf.R = np.diag([0.1, 0.1])  # 观测噪声协方差
    ukf.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # 过程噪声协方差
    return ukf

# 生成真实轨迹
true_trajectory = generate_true_trajectory(initial_state, dt, num_steps)



camera_rpy_to=[0, 45, -90]
camera_rpy_back=[45, 0, 90]
rotation_matrix_to=rotate_matrix_coordinate(camera_rpy_to[0],camera_rpy_to[1],camera_rpy_to[2])
rotation_matrix_back=rotate_matrix_coordinate(camera_rpy_back[0],camera_rpy_back[1],camera_rpy_back[2])


# 生成带噪声的观测数据
# 生成带噪声的观测数据
noise_std = 0.5  # 观测噪声标准差
observations = generate_noisy_observations(true_trajectory, rotation_matrix_to, np.zeros(3),H, noise_std)
# 初始化 UKF
print(observations)
ukf = initialize_ukf(initial_state, dt,rotation_matrix_to,np.zeros(3))

# 运行 UKF 进行状态估计
estimated_trajectory = []
for z in observations:
    ukf.predict()
    ukf.update(z)
    estimated_trajectory.append(ukf.x[:3])  # 仅保存位置估计
estimated_trajectory = np.array(estimated_trajectory)

# 使用 plotly 绘制轨迹
fig = go.Figure()

# 真实轨迹
fig.add_trace(go.Scatter3d(
    x=true_trajectory[:, 0], y=true_trajectory[:, 1], z=true_trajectory[:, 2],
    mode='lines+markers', name='True Trajectory',
    line=dict(color='blue', width=2)
))

# 估计轨迹
fig.add_trace(go.Scatter3d(
    x=estimated_trajectory[:, 0], y=estimated_trajectory[:, 1], z=estimated_trajectory[:, 2],
    mode='lines+markers', name='Estimated Trajectory',
    line=dict(color='red', width=2)
))

# # 观测数据
# fig.add_trace(go.Scatter3d(
#     x=observations[:, 0], y=observations[:, 1], z=np.zeros_like(observations[:, 0]),
#     mode='markers', name='Observations',
#     marker=dict(color='green', size=4)
# ))

# 设置图形布局
fig.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)'
    ),
    title='3D Trajectory Estimation using UKF'
)

# 显示图形
fig.show()