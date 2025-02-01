import numpy as np
import plotly.graph_objects as go

ball_memory=np.load("ball_pixels.npy")
def ekf_predict(ball_memory):
    if ball_memory.shape[0]<300:
        return None
    # 1) 反转数组
    arr_rev = ball_memory[::-1]

    # 2) 对反转后的数组在前两列做 unique，并获取返回的索引（对应反转后的坐标）
    _, idx_rev = np.unique(arr_rev[:, :2], axis=0, return_index=True)

    # 3) 将反转索引转回在原数组中的位置
    idx_original = len(arr_rev) - 1 - idx_rev

    # 4) 排序（让结果按原数组从上到下的顺序）
    idx_sorted = np.sort(idx_original)

    # 最终结果
    # result = ball_memory[idx_sorted]
    ball_memory=ball_memory[idx_sorted]
    ball_memory=ball_memory[:,:3]
    # import numpy as np
    # import numpy as np

    # 定义参数
    # dt = 0.1  # 时间步长
    # print(ball_memory)
    g = 9.81  # 重力加速度

    # 相机内参矩阵（假设已知）
    cy = 640
    cx = 360
    fx = 673.797
    fy = 673.797
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 状态初始化 [x, y, z, vx, vy, vz]
    # x = np.array([3, 0, 2, -6, 0, 1]).reshape(-1, 1)
    cov_P = np.eye(6)  # 初始协方差矩阵
    cov_Q = np.diag([1, 1, 1, 1, 1, 1])   # 过程噪声协方差
    cov_O = np.diag([1, 1])  # 观测噪声协方差

    # 定义相机坐标旋转矩阵
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

    # 定义观测模型
    def h(x):

        """将三维位置投影到二维像素坐标系"""

        u = fx * x[0] / x[2] + cx
        v = fy * x[1] / x[2] + cy
        return np.array([u, v]).reshape(-1, 1)
    def h_rotation(x,R,T=np.zeros(3)):

        """将三维位置投影到二维像素坐标系"""
        # print(x)
        # 提取状态变量
        x_w, y_w, z_w = x[0, 0], x[1, 0], x[2, 0]
        # 转换到相机坐标系
        world_point = np.array([x_w, y_w, z_w])
        # print("h", world_point)
        cam_point = R @ world_point + T
        # print(cam_point)
        x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]
        # print(x_c, y_c, z_c)
        # print((x_c, y_c, z_c))
        u = fx * x_c / z_c + cx
        v = fy * y_c / z_c + cy
        # print(u)
        return np.array([u, v]).reshape(-1, 1)
    # def h_p(x)
    def estimate_state(u,v,z_c,vu,vv,R_back,T=np.zeros(3)):
        # print(vu,vv)
        x_c = (u-cx)*z_c/fx
        y_c = (v-cy)*z_c/fy
        vx_c=vu*z_c/fx
        vy_c=vv*z_c/fy
        camera_point = np.array([x_c, y_c, z_c])
        world_point = R_back@camera_point - T
        camera_velocity = np.array([vx_c, vy_c,0.7*vy_c])
        world_velocity=R_back@camera_velocity - T
        return world_point,world_velocity
    def jacobian_H(x):
        """计算观测模型的雅可比矩阵"""

        H = np.zeros((2, 6))
        H[0, 0] = fx / x[2]
        H[0, 2] = -fx * x[0] / (x[2]**2)
        H[1, 1] = fy / x[2]
        H[1, 2] = -fy * x[1] / (x[2]**2)
        return H
    def compute_jacobian(x,R,T=np.zeros(3)):
        # print(x)
        # 提取状态变量
        x_w, y_w, z_w = x[0,0], x[1,0], x[2,0]
        # 转换到相机坐标系
        world_point = np.array([x_w, y_w, z_w])
        # print("w",world_point)
        cam_point = R @ world_point + T
        # print(cam_point)
        x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]
        # print(x_c, y_c, z_c)
        # 计算偏导数
        H = np.zeros((2, 6))
        H[0, 0] = (fx / z_c) * R[0, 0] - (fx * x_c / z_c**2) * R[2, 0]
        H[0, 1] = (fx / z_c) * R[0, 1] - (fx * x_c / z_c**2) * R[2, 1]
        H[0, 2] = (fx / z_c) * R[0, 2] - (fx * x_c / z_c**2) * R[2, 2]
        H[1, 0] = (fy / z_c) * R[1, 0] - (fy * y_c / z_c**2) * R[2, 0]
        H[1, 1] = (fy / z_c) * R[1, 1] - (fy * y_c / z_c**2) * R[2, 1]
        H[1, 2] = (fy / z_c) * R[1, 2] - (fy * y_c / z_c**2) * R[2, 2]
        return H
    # 求解二次方程的函数
    def solve_plane_intersection(x):
        # print(x)
        t=abs(x[0]/x[3])
        # print(t)
        return t

    # 预测球经过平面的坐标
    def predict_intersection(x):
        t = solve_plane_intersection(x)
        # print(x,t)
        # 计算经过平面时的位置
        x_intersect = x[0] + x[3] * t
        y_intersect = x[1] + x[4] * t
        z_intersect = x[2] + x[5] * t - 0.5 * g * t**2
        return np.array([x_intersect, y_intersect, z_intersect])

    camera_rpy_to=[0, 45, -90]
    camera_rpy_back=[45, 0, 90]
    rotation_matrix_to=rotate_matrix_coordinate(camera_rpy_to[0],camera_rpy_to[1],camera_rpy_to[2])
    rotation_matrix_back=rotate_matrix_coordinate(camera_rpy_back[0],camera_rpy_back[1],camera_rpy_back[2])
    initial_z = fx / 20 * 0.055

    # initial_z = fx / 50 * 0.55
    estimated_trajectory = []
    impact_points = []
    initial_dt = ball_memory[1][2] - ball_memory[0][2]
    initial_u,initial_v=ball_memory[1][0],ball_memory[1][1]
    initial_vu=(ball_memory[1][0]-ball_memory[0][0])/initial_dt
    initial_vv=(ball_memory[1][1]-ball_memory[0][1])/initial_dt
    world_point,world_velocity=estimate_state(initial_u,initial_v,initial_z,initial_vu,initial_vv,rotation_matrix_back)
    # print(world_point,world_velocity)

    x = np.array([world_point[0], world_point[1], world_point[2], world_velocity[0], world_velocity[1], world_velocity[2]]).reshape(-1, 1)
    # print(x)
    # EKF 主循环
    for step in range(1,ball_memory.shape[0]):
        dt=ball_memory[step][2]-ball_memory[step-1][2]
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, -g * dt, 0, 0, 1]
        ])
        # 预测步骤
        x = F @ x
        cov_P = F @ cov_P @ F.T + cov_Q

        # 获取观测数据
        z=ball_memory[step,:2].reshape(-1, 1)
        # print(z)
        # 计算观测模型的雅可比矩阵和预测的观测值
        H = compute_jacobian(x,rotation_matrix_to)
        z_pred = h_rotation(x,rotation_matrix_to)
        # 更新步骤
        S = H @ cov_P @ H.T + cov_O  # 创新协方差
        K = cov_P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
        # print(z, z_pred, x)
        y = z - z_pred  # 创新
        impact_point=predict_intersection(x)
        impact_points.append(impact_point.reshape(1,-1).squeeze())
        x = x + K @ y  # 更新状态
        cov_P = (np.eye(6) - K @ H) @ cov_P  # 更新协方差

        # 保存估计位置
        estimated_trajectory.append(x[:3].flatten())

        estimated_trajectory = np.array(estimated_trajectory)
    return impact_points[-1]

