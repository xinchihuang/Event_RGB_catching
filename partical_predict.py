import numpy as np
import plotly.graph_objects as go

# 参数设置
g = 9.8  # 重力加速度 (m/s^2)
dt = 0.01  # 时间间隔 (s)
num_steps = 100  # 时间步数
num_particles = 10000  # 粒子数量
process_noise_std = 0.1  # 过程噪声标准差
observation_noise_std = 0.5  # 观测噪声标准差

# 初始状态 [X, Y, Z, Vx, Vy, Vz]
initial_state = np.array([0, 0, 0, 10, 0, 0])

# 生成真实轨迹
def generate_true_trajectory(initial_state, dt, num_steps):
    t = np.arange(0, num_steps * dt, dt)
    X = initial_state[0] + initial_state[3] * t
    Y = initial_state[1] + initial_state[4] * t
    Z = initial_state[2] + initial_state[5] * t - 0.5 * g * t**2
    return np.vstack([X, Y, Z]).T

# 生成带噪声的观测数据
def generate_noisy_observations(true_trajectory, noise_std):
    observations = true_trajectory[:, :2]  # 仅观测 X 和 Y
    noise = np.random.normal(0, noise_std, observations.shape)
    return observations + noise

# 状态转移函数（考虑重力的抛物线运动模型）
def state_transition(particles, dt):
    # particles: [num_particles, 6]
    X, Y, Z, Vx, Vy, Vz = particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], particles[:, 4], particles[:, 5]
    X_new = X + Vx * dt
    Y_new = Y + Vy * dt
    Z_new = Z + Vz * dt - 0.5 * g * dt**2
    Vx_new = Vx
    Vy_new = Vy
    Vz_new = Vz - g * dt
    # 添加过程噪声
    noise = np.random.normal(0, process_noise_std, particles.shape)
    return np.vstack([X_new, Y_new, Z_new, Vx_new, Vy_new, Vz_new]).T + noise

# 观测函数（假设仅观测 X 和 Y）
def observation_model(particles):
    # particles: [num_particles, 6]
    return particles[:, :2]  # 仅返回 X 和 Y

# 计算粒子权重
def compute_weights(particles, observation, noise_std):
    # particles: [num_particles, 6]
    # observation: [2,]
    predicted_observation = observation_model(particles)  # [num_particles, 2]
    error = observation - predicted_observation  # [num_particles, 2]
    # 计算每个粒子的观测概率（高斯分布）
    weights = np.exp(-0.5 * np.sum(error**2, axis=1) / noise_std**2)
    weights /= np.sum(weights)  # 归一化
    return weights

# 重采样
def resample(particles, weights):
    indices = np.random.choice(np.arange(len(particles)), size=len(particles), p=weights)
    return particles[indices]

# 初始化粒子
def initialize_particles(num_particles, initial_state, initial_std):
    # initial_state: [6,]
    # initial_std: [6,]
    particles = np.random.normal(initial_state, initial_std, (num_particles, 6))
    return particles

# 生成真实轨迹
true_trajectory = generate_true_trajectory(initial_state, dt, num_steps)

# 生成带噪声的观测数据
observations = generate_noisy_observations(true_trajectory, observation_noise_std)

# 初始化粒子
initial_std = np.array([10, 10, 10, 5, 5, 5])  # 初始猜测的不确定性
particles = initialize_particles(num_particles, initial_state, initial_std)

# 运行粒子滤波
estimated_trajectory = []
for z in observations:
    # 预测步骤
    particles = state_transition(particles, dt)
    # 更新步骤
    weights = compute_weights(particles, z, observation_noise_std)
    # 重采样
    particles = resample(particles, weights)
    # 状态估计（加权平均）
    estimated_state = np.average(particles, axis=0, weights=weights)
    estimated_trajectory.append(estimated_state[:3])  # 仅保存位置估计
estimated_trajectory = np.array(estimated_trajectory)

# 使用 plotly 绘制三维轨迹
fig = go.Figure()

# 真实轨迹
fig.add_trace(go.Scatter3d(
    x=true_trajectory[:, 0], y=true_trajectory[:, 1], z=true_trajectory[:, 2],
    mode='lines', name='True Trajectory',
    line=dict(color='blue', width=2)
))

# 估计轨迹
fig.add_trace(go.Scatter3d(
    x=estimated_trajectory[:, 0], y=estimated_trajectory[:, 1], z=estimated_trajectory[:, 2],
    mode='lines', name='Estimated Trajectory',
    line=dict(color='red', width=2)
))

# 观测数据（假设 Z=0）
fig.add_trace(go.Scatter3d(
    x=observations[:, 0], y=observations[:, 1], z=np.zeros_like(observations[:, 0]),
    mode='markers', name='Observations',
    marker=dict(color='green', size=4)
))

# 设置图形布局
fig.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)'
    ),
    title='3D Trajectory Estimation using Particle Filter'
)

# 显示图形
fig.show()