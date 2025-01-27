import numpy as np
import plotly.graph_objects as go

ball_memory=np.load("ball_memory.npy")
# import numpy as np


g = 9.81  # 重力加速度

# 相机内参矩阵（假设已知）
cx = 619.674
cy = 333.682
fx = 673.797
fy = 673.797
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 状态初始化 [x, y, z, vx, vy, vz]
x = np.array([0, 0, 10, 5, 0, 0]).reshape(-1, 1)
P = np.eye(6)  # 初始协方差矩阵
Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # 过程噪声协方差
R = np.diag([5, 5])  # 观测噪声协方差

# 定义状态转移矩阵


# 定义观测模型
def h(x):

    """将三维位置投影到二维像素坐标系"""
    if len(x)==2:
        u = fx * x[0] / 1 + cx
        v = fy * x[1] / 1 + cy
    else:
        u = fx * x[0] / x[2] + cx
        v = fy * x[1] / x[2] + cy
    return np.array([u, v]).reshape(-1, 1)
# def h_p(x)

def jacobian_H(x):
    """计算观测模型的雅可比矩阵"""
    # print(x)
    H = np.zeros((2, 6))
    H[0, 0] = fx / x[2]
    H[0, 2] = -fx * x[0] / (x[2]**2)
    H[1, 1] = fy / x[2]
    H[1, 2] = -fy * x[1] / (x[2]**2)
    return H
estimated_trajectory = []
observed_point=[]
# EKF 主循环
for step in range(1,ball_memory.shape[0]):

    dt=ball_memory[step][2]-ball_memory[step-1][2]
    F = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, -np.cos(np.pi)*g * dt, 0, 0, 1, 0],
        [0, 0, -np.cos(np.pi)*g * dt, 0, 0, 1]
    ])
    # 预测步骤
    x = F @ x
    P = F @ P @ F.T + Q

    # 获取观测数据
    # z = ball_memory[step,:2]
    # print(z)
    observed_point.append(ball_memory[step])
    z=h(ball_memory[step,:2])

    # 计算观测模型的雅可比矩阵和预测的观测值
    H = jacobian_H(x)
    z_pred = h(x)

    # 更新步骤
    S = H @ P @ H.T + R  # 创新协方差
    K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
    y = z - z_pred  # 创新
    x = x + K @ y  # 更新状态
    P = (np.eye(6) - K @ H) @ P  # 更新协方差

    # 保存估计位置
    estimated_trajectory.append(x[:3].flatten())
    # 打印状态和观测
    print(f"Step {step + 1}")
    print(f"Estimated state:\n{x.ravel()}")
    print(f"Measurement:\n{z.ravel()}\n")
# true_trajectory = np.array(true_trajectory)
estimated_trajectory = np.array(estimated_trajectory)

# 使用 Plotly 绘制轨迹
fig = go.Figure()

# # 添加真实轨迹
# fig.add_trace(go.Scatter3d(
#     x=true_trajectory[:, 0],
#     y=true_trajectory[:, 1],
#     z=true_trajectory[:, 2],
#     mode='lines+markers',
#     name='True Trajectory',
#     line=dict(color='blue', width=2),
#     marker=dict(size=4)
# ))
print(observed_point)
fig.add_trace(go.Scatter(
    x=observed_point[:, 0],
    y=observed_point[:, 1],
    mode='markers',
    name='Observed point',
    marker=dict(color='red', size=6, symbol='circle')
))
# 添加估计轨迹
fig.add_trace(go.Scatter3d(
    x=estimated_trajectory[:, 0],
    y=estimated_trajectory[:, 1],
    z=estimated_trajectory[:, 2],
    mode='lines+markers',
    name='Estimated Trajectory',
    line=dict(color='red', width=2),
    marker=dict(size=4)
))

# 配置布局
fig.update_layout(
    title='3D Trajectory: True vs Estimated',
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()





