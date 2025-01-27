import numpy as np
import plotly.graph_objects as go

# ball_memory=np.array([[-0.72848374,  0.63488865,  0.       ],
#                      [-0.74531933,  0.60499671,  0.0016253 ],
#                      [-0.74531933,  0.60499671,  0.00203371],
#                      [-0.77473359,  0.52431068,  0.00798178],
#                      [-0.77473359,  0.52431068,  0.00843167],
#                      [-0.79365686,  0.49659109,  0.00945067],
#                      [-0.79365686,  0.49659109,  0.00983453],
#                      [-0.79365686,  0.49659109,  0.01027226],
#                      [-0.83374014,  0.44223946,  0.01261377],
#                      [-0.85502573,  0.41547656,  0.01401258],
#                      [-0.85502573,  0.41547656,  0.01438212],
#                      [-0.87723895,  0.38890411,  0.01545906],
#                      [-0.87723895,  0.38890411,  0.01584601],
#                      [-0.90900193,  0.33377445,  0.0167985 ],
#                      [-0.92029198,  0.31552452,  0.01788568],
#                      [-0.97126097,  0.25272527,  0.01985264],
#                      [-0.97126097,  0.25272527,  0.02024579],
#                      [-1.05041663,  0.2110932 ,  0.02155709],
#                      [-1.05041663,  0.2110932 ,  0.02193356],
#                      [-1.08022898,  0.16325133,  0.02292275],
#                      [-1.1528435 ,  0.10656134,  0.02426267],
#                      [-1.21298599,  0.05677011,  0.0254643 ],
#                      [-1.27883478, -0.02784242, 0.02689981]])
ball_memory=np.load("ball_memory.npy")
# import numpy as np
# import numpy as np

# 定义参数
# dt = 0.1  # 时间步长
print(ball_memory)
g = 9.81  # 重力加速度

# 相机内参矩阵（假设已知）
cy = 640
cx = 360
fx = 500
fy = 500
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 状态初始化 [x, y, z, vx, vy, vz]
x = np.array([0, 0, 4, 0, 0, 0]).reshape(-1, 1)
P = np.eye(6)  # 初始协方差矩阵
Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1])   # 过程噪声协方差
R = np.diag([0.1, 0.1])  # 观测噪声协方差

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

    H = np.zeros((2, 6))
    H[0, 0] = fx / x[2]
    H[0, 2] = -fx * x[0] / (x[2]**2)
    H[1, 1] = fy / x[2]
    H[1, 2] = -fy * x[1] / (x[2]**2)
    return H
estimated_trajectory = []
observed_point = []
# EKF 主循环
for step in range(1,ball_memory.shape[0]):

    dt=ball_memory[step][2]-ball_memory[step-1][2]
    F = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, np.cos(np.pi/4) * dt, 0, 0, 1, 0],
        [0, 0, -np.sin(np.pi/4)* dt, 0, 0, 1]
    ])
    # 预测步骤
    x = F @ x
    P = F @ P @ F.T + Q

    # 获取观测数据
    # z = ball_memory[step,:2]
    # print(z)
    observed_point.append([ball_memory[step][0],ball_memory[step][1],1])
    # print(ball_memory[step,:2])
    z=h(ball_memory[step,:2])
    # z=ball_memory[step,:2].reshape(-1,1)
    # print(z)
    # 计算观测模型的雅可比矩阵和预测的观测值
    H = jacobian_H(x)
    z_pred = h(x)
    # print(z_pred)

    # 更新步骤
    S = H @ P @ H.T + R  # 创新协方差
    K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
    y = z - z_pred  # 创新
    x = x + K @ y  # 更新状态
    P = (np.eye(6) - K @ H) @ P  # 更新协方差

    # 保存估计位置
    estimated_trajectory.append(x[:3].flatten())
    # 打印状态和观测
    # print(f"Step {step + 1}")
    # print(f"Estimated state:\n{x.ravel()}")
    # print(f"Measurement:\n{z.ravel()}\n")
# true_trajectory = np.array(true_trajectory)
estimated_trajectory = np.array(estimated_trajectory)
observed_point = np.array(observed_point)

camera_point =np.array([[0,0,0]])
theta = -np.pi/4  # 45 度
# theta = 0

# 构建旋转矩阵
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
estimated_trajectory=estimated_trajectory@R_x
observed_point=observed_point@R_x
camera_point=camera_point@R_x

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
fig.add_trace(go.Scatter3d(
    x=observed_point[:, 0],
    y=observed_point[:, 1],
    z=observed_point[:, 2],
    mode='markers',
    name='Observed point',
    marker=dict(color='blue', size=6, symbol='circle')
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
fig.add_trace(go.Scatter3d(
    x=camera_point[:, 0],
    y=camera_point[:, 1],
    z=camera_point[:, 2],
    mode='lines+markers',
    name='Camera',
    line=dict(color='green', width=2),
    marker=dict(size=10)
))
# 配置布局
fig.update_layout(
    title='3D Trajectory: Estimated',
    scene=dict(
        xaxis=dict(
            title='X Axis',
            tickwidth=10,  # 设置刻度线的粗细
            linewidth=10,  # 设置坐标轴线条的粗细4tickfont=dict(size=14),  # 设置刻度字体大小
            tickfont=dict(size=20),  # 设置刻度字体大小
            titlefont=dict(size=30),  # 设置坐标轴标题字体大小
        ),
        yaxis=dict(
            title='Y Axis',
            tickwidth=10,
            linewidth=10,
            tickfont=dict(size=20),  # 设置刻度字体大小
            titlefont=dict(size=30),  # 设置坐标轴标题字体大小
        ),
        zaxis=dict(
            title='Z Axis',
            tickwidth=10,
            linewidth=10,
            tickfont=dict(size=20),  # 设置刻度字体大小
            titlefont=dict(size=30),  # 设置坐标轴标题字体大小
        ),
       # 确保三轴比例相同
    ),
    # margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()





