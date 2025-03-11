import random
import numpy as np
from scipy.optimize import least_squares
from utils import transform_coordinate,rotate_matrix_coordinate,pixel_to_world
class PixelSizeFitter:
    def __init__(self,camera_params,rotation_matrix,ball_diameter=0.14):
        self.camera_params = camera_params
        self.rotation_matrix = rotation_matrix
    def set_data(self,data):
        self.data = data
    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix)
        X0, Y0, Z0 = world_coords[0]
        initial_guess = [X0, Y0, Z0, 0, 0, 0]  # [X0, Y0, Z0, Vx, Vy, Vz]
        timesteps = self.data[:, 3]
        def trajectory_model(params, t):
            X0, Y0, Z0, Vx, Vy, Vz = params
            X_pred = X0 + Vx * t
            Y_pred = Y0 + Vy * t
            Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
            return np.column_stack([X_pred, Y_pred, Z_pred])

        def residuals(params):
            predicted_coords = trajectory_model(params, timesteps)
            return (world_coords - predicted_coords).ravel()  # 展平成1D数组

        opt_result = least_squares(
            residuals,
            initial_guess,
            max_nfev=200,
            xtol=1e-6,
            ftol=1e-6,
            verbose=0
        )
        success = opt_result.success
        params = opt_result.x.tolist() if success else None
        avg_error = np.mean(np.abs(opt_result.fun)) if success else np.inf
        predeicted_params = {
            'success': success,
            "points": world_coords,
            'params': {
                'x0': params[0],
                'y0': params[1],
                'z0': params[2],
                'vx': params[3],
                'vy': params[4],
                'vz': params[5],
            } if success else None,
            'cost': opt_result.cost,
            'reprojection_error': avg_error,
            'iterations': opt_result.nfev
        }
        self.predeicted_params = predeicted_params
        return predeicted_params
    def predict(self):
        x0 = self.predeicted_params['params']['x0']
        y0 = self.predeicted_params['params']['y0']
        z0 = self.predeicted_params['params']['z0']
        vx = self.predeicted_params['params']['vx']
        vy = self.predeicted_params['params']['vy']
        vz = self.predeicted_params['params']['vz']
        impact_t = abs(x0 / vx)
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point=[0,impact_y,impact_z]
        return impact_point
class MonoEKF:
    def __init__(self,camera_params,rotation_matrix):
        self.camera_params = camera_params
        self.rotation_matrix = rotation_matrix

    def set_data(self,data):
        self.data = data
    def get_params(self):
        fx, fy = self.camera_params['fx'], self.camera_params['fy']
        cx, cy = self.camera_params['cx'], self.camera_params['cy']
        dist_coeffs = self.camera_params.get('dist_coeffs', np.zeros(5))
        k1, k2, p1, p2, k3 = dist_coeffs if len(dist_coeffs) == 5 else np.append(dist_coeffs,
                                                                                 [0] * (5 - len(dist_coeffs)))
        g = 9.81
        cov_P = np.eye(6)  # 初始协方差矩阵
        cov_Q = np.diag([1, 1, 1, 1, 1, 1])  # 过程噪声协方差
        cov_O = np.diag([1, 1])  # 观测噪声协方差
        initial_z=5

        def h_rotation(x, R, T=np.zeros(3)):
            """将三维位置投影到二维像素坐标系"""
            # 提取状态变量
            x_w, y_w, z_w = x[0, 0], x[1, 0], x[2, 0]
            # 转换到相机坐标系
            world_point = np.array([x_w, y_w, z_w])
            cam_point = R @ world_point + T
            x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]
            u = fx * x_c / z_c + cx
            v = fy * y_c / z_c + cy
            return np.array([u, v]).reshape(-1, 1)

        def estimate_state(u, v, z_c, vu, vv, R_back, T=np.zeros(3)):
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy
            vx_c = vu * z_c / fx
            vy_c = vv * z_c / fy
            camera_point = np.array([x_c, y_c, z_c])
            world_point = R_back @ camera_point - T
            camera_velocity = np.array([vx_c, vy_c, 1.2 * vy_c])
            world_velocity = R_back @ camera_velocity - T
            return world_point, world_velocity
        def compute_jacobian(x, R, T=np.zeros(3)):
            # print(x)
            # 提取状态变量
            x_w, y_w, z_w = x[0, 0], x[1, 0], x[2, 0]
            # 转换到相机坐标系
            world_point = np.array([x_w, y_w, z_w])
            # print("w",world_point)
            cam_point = R @ world_point + T
            # print(cam_point)
            x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]
            # print(x_c, y_c, z_c)
            # 计算偏导数
            H = np.zeros((2, 6))
            H[0, 0] = (fx / z_c) * R[0, 0] - (fx * x_c / z_c ** 2) * R[2, 0]
            H[0, 1] = (fx / z_c) * R[0, 1] - (fx * x_c / z_c ** 2) * R[2, 1]
            H[0, 2] = (fx / z_c) * R[0, 2] - (fx * x_c / z_c ** 2) * R[2, 2]
            H[1, 0] = (fy / z_c) * R[1, 0] - (fy * y_c / z_c ** 2) * R[2, 0]
            H[1, 1] = (fy / z_c) * R[1, 1] - (fy * y_c / z_c ** 2) * R[2, 1]
            H[1, 2] = (fy / z_c) * R[1, 2] - (fy * y_c / z_c ** 2) * R[2, 2]
            return H
        ball_memory=self.data
        rotation_matrix=self.rotation_matrix
        estimated_trajectory = []
        initial_dt = ball_memory[1][2] - ball_memory[0][2]
        initial_u, initial_v = ball_memory[1][0], ball_memory[1][1]
        initial_vu = (ball_memory[1][0] - ball_memory[0][0]) / initial_dt
        initial_vv = (ball_memory[1][1] - ball_memory[0][1]) / initial_dt
        world_point, world_velocity = estimate_state(initial_u, initial_v, initial_z, initial_vu, initial_vv,
                                                     rotation_matrix)
        x = np.array([world_point[0], world_point[1], world_point[2], world_velocity[0], world_velocity[1],
                      world_velocity[2]]).reshape(-1, 1)
        for step in range(1, ball_memory.shape[0]):
            dt = ball_memory[step][2] - ball_memory[step - 1][2]
            F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, -g * dt, 0, 0, 1]
            ])
            x = F @ x
            cov_P = F @ cov_P @ F.T + cov_Q
            z = ball_memory[step, :2].reshape(-1, 1)
            # 计算观测模型的雅可比矩阵和预测的观测值
            H = compute_jacobian(x, rotation_matrix)
            z_pred = h_rotation(x, rotation_matrix)
            # print(z,z_pred)

            # 更新步骤
            S = H @ cov_P @ H.T + cov_O  # 创新协方差
            K = cov_P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
            # print(z, z_pred, x)
            y = z - z_pred  # 创新
            x = x + K @ y  # 更新状态
            cov_P = (np.eye(6) - K @ H) @ cov_P  # 更新协方差

            estimated_trajectory.append(x[:3].flatten())

        estimated_trajectory = np.array(estimated_trajectory)
        predeicted_params = {
            "points": estimated_trajectory,
            'params': {
                'x0': x[0],
                'y0': x[1],
                'z0': x[2],
                'vx': x[3],
                'vy': x[4],
                'vz': x[5],
            },
        }
        self.predeicted_params = predeicted_params
        return predeicted_params

    def predict(self):
        x0 = self.predeicted_params['params']['x0']
        y0 = self.predeicted_params['params']['y0']
        z0 = self.predeicted_params['params']['z0']
        vx = self.predeicted_params['params']['vx']
        vy = self.predeicted_params['params']['vy']
        vz = self.predeicted_params['params']['vz']
        impact_t = abs(x0 / vx)
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        return impact_point
class MonoEKF3D:
    def __init__(self,camera_params,rotation_matrix_to,rotation_matrix_back,ball_diameter=0.14):
        self.camera_params = camera_params
        self.rotation_matrix_to = rotation_matrix_to
        self.rotation_matrix_back = rotation_matrix_back
        self.ball_diameter = ball_diameter

    def set_data(self,data):
        self.data = data
    def get_params(self):
        # 内部测量模型：预测像素坐标和球体直径
        def h_measurement(x, R, ball_diameter, T=np.zeros(3)):
            """
            将状态向量投影到测量空间 [u, v, d]
            :param x: 状态向量 [x, y, z, vx, vy, vz] (6x1向量)
            :param R: 旋转矩阵
            :param ball_diameter: 球体真实直径
            :param T: 平移向量（默认0）
            :return: 预测测量向量 [u, v, d] (3x1向量)
            """
            # 提取世界坐标
            x_w, y_w, z_w = x[0, 0], x[1, 0], x[2, 0]
            world_point = np.array([x_w, y_w, z_w])
            cam_point = R @ world_point + T
            x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]
            u = fx * x_c / z_c + cx
            v = fy * y_c / z_c + cy
            # 根据透视投影计算直径预测 d_pred = fx * D_real / z_c
            d_pred = fx * ball_diameter / z_c
            return np.array([u, v, d_pred]).reshape(-1, 1)

        # 内部函数：利用像素信息和深度，估计初始状态
        def estimate_state(u, v, z_c, vu, vv, vz, R_back, T=np.zeros(3)):
            print(u, v, z_c, vu, vv, vz)
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy
            vx_c = vu * z_c / fx
            vy_c = vv * z_c / fy
            vz_c = vz * z_c / fx
            print(vx_c,vy_c,vz_c)
            camera_point = np.array([x_c, y_c, z_c])
            world_point = R_back @ camera_point - T
            # 这里对速度进行了一个简单处理，可以根据需要调整
            camera_velocity = np.array([vx_c, vy_c, vz_c])
            world_velocity = R_back @ camera_velocity - T
            return world_point, world_velocity

        # 内部函数：计算观测模型的雅可比矩阵（3x6），对应 [u, v, d]
        def compute_jacobian_full(x, R, ball_diameter, T=np.zeros(3)):
            x_w, y_w, z_w = x[0, 0], x[1, 0], x[2, 0]
            world_point = np.array([x_w, y_w, z_w])
            cam_point = R @ world_point + T
            x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]

            H = np.zeros((3, 6))
            # 对 u 的偏导数
            H[0, 0] = (fx / z_c) * R[0, 0] - (fx * x_c / (z_c ** 2)) * R[2, 0]
            H[0, 1] = (fx / z_c) * R[0, 1] - (fx * x_c / (z_c ** 2)) * R[2, 1]
            H[0, 2] = (fx / z_c) * R[0, 2] - (fx * x_c / (z_c ** 2)) * R[2, 2]
            # 对 v 的偏导数
            H[1, 0] = (fy / z_c) * R[1, 0] - (fy * y_c / (z_c ** 2)) * R[2, 0]
            H[1, 1] = (fy / z_c) * R[1, 1] - (fy * y_c / (z_c ** 2)) * R[2, 1]
            H[1, 2] = (fy / z_c) * R[1, 2] - (fy * y_c / (z_c ** 2)) * R[2, 2]
            # 第三行：直径 d 的偏导数，仅依赖于位置分量
            factor = -fx * ball_diameter / (z_c ** 2)
            H[2, 0] = factor * R[2, 0]
            H[2, 1] = factor * R[2, 1]
            H[2, 2] = factor * R[2, 2]
            # 速度部分对测量没有影响，保持为0
            return H
        # 提取相机参数
        fx, fy = self.camera_params['fx'], self.camera_params['fy']
        cx, cy = self.camera_params['cx'], self.camera_params['cy']
        dist_coeffs = self.camera_params.get('dist_coeffs', np.zeros(5))
        if len(dist_coeffs) != 5:
            dist_coeffs = np.append(dist_coeffs, [0] * (5 - len(dist_coeffs)))
        # 重力加速度
        g = 9.81

        # 初始化滤波器协方差
        cov_P = np.eye(6)  # 状态协方差矩阵，状态为 [x, y, z, vx, vy, vz]
        cov_Q = np.diag([1, 1, 1, 1, 1, 1])  # 过程噪声协方差
        # 观测噪声协方差，3x3矩阵，这里需要根据实际情况设置标准差
        sigma_u, sigma_v, sigma_d = 1, 1, 1  # 示例数值，可根据实际情况调整
        cov_O = np.diag([sigma_u ** 2, sigma_v ** 2, sigma_d ** 2])

        ball_memory = self.data

        # 根据第一帧观测的球体直径 d 估计初始深度
        # 透视投影公式: d_image = fx * D_real / z  =>  z = fx * D_real / d_image
        # print(ball_memory[0][2])
        observed_diameter = ball_memory[0][2]  # 第一帧图像中球体的直径（像素）
        initial_z = (fx * self.ball_diameter) / observed_diameter

        # 估计轨迹列表，用于记录每个时间步的状态位置
        estimated_trajectory = []

        # 利用第一帧和第二帧数据初始化状态
        initial_dt = ball_memory[1][3] - ball_memory[0][3]
        initial_u, initial_v = ball_memory[1][0], ball_memory[1][1]
        initial_vu = (ball_memory[1][0] - ball_memory[0][0]) / initial_dt
        initial_vv = (ball_memory[1][1] - ball_memory[0][1]) / initial_dt
        initial_vz = -(ball_memory[1][2] - ball_memory[0][2]) / initial_dt
        world_point, world_velocity = estimate_state(initial_u, initial_v, initial_z, initial_vu, initial_vv, initial_vz, self.rotation_matrix_back)
        x = np.array([world_point[0], world_point[1], world_point[2],
                      world_velocity[0], world_velocity[1], world_velocity[2]]).reshape(-1, 1)
        # print(x)
        # 迭代更新：利用 EKF 更新状态
        for step in range(1, ball_memory.shape[0]):
            dt = ball_memory[step][3] - ball_memory[step - 1][3]
            # 状态转移矩阵 F（考虑重力对 z 轴的影响）
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

            # 观测向量 z = [u, v, d] (3x1)
            z = ball_memory[step, :3].reshape(-1, 1)
            # 计算测量雅可比矩阵 H (3x6)
            H = compute_jacobian_full(x, self.rotation_matrix_to, self.ball_diameter)
            # 预测测量值
            z_pred = h_measurement(x, self.rotation_matrix_to, self.ball_diameter)
            # 创新协方差
            S = H @ cov_P @ H.T + cov_O
            # 卡尔曼增益
            K = cov_P @ H.T @ np.linalg.inv(S)
            # 创新（残差）
            y = z - z_pred
            # 更新状态
            x = x + K @ y
            # 更新状态协方差
            cov_P = (np.eye(6) - K @ H) @ cov_P

            # 保存当前时刻预测的三维位置
            estimated_trajectory.append(x[:3].flatten())
        print(x)
        estimated_trajectory = np.array(estimated_trajectory)
        predeicted_params = {
            "points": estimated_trajectory,
            "params": {
                "x0": x[0, 0],
                "y0": x[1, 0],
                "z0": x[2, 0],
                "vx": x[3, 0],
                "vy": x[4, 0],
                "vz": x[5, 0],
            },
        }
        self.predeicted_params = predeicted_params
        return predeicted_params


    def predict(self):
        params = self.predeicted_params['params']
        x0 = params['x0']
        y0 = params['y0']
        z0 = params['z0']
        vx = params['vx']
        vy = params['vy']
        vz = params['vz']
        # 碰撞时间：假设球体以 vx 匀速运动到 x=0
        impact_t = abs(x0 / vx)
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        return impact_point
class ParabolaFitterOld:
    def __init__(self,robot_pose=[0,0,0],camera_rpy_to=[0,0,0],camera_rpy_back=[0,0,0],min_data_samples=30):

        self.camera_pose=robot_pose
        # This rpy indicates the transformation of a vector from world coordinate to the camera coordinate
        self.camera_rpy_to=camera_rpy_to
        # This rpy indicates the transformation of a vector from camera coordinate to the world coordinate
        self.camera_rpy_back = camera_rpy_back
        self.g=-9.81
        self.min_date_sample=min_data_samples


    def basic_fit(self):

        pass
    def is_inlier(self,data_points,model_points,threshold):
        """
        Determine if a model predicted point is inlier
        Args:
            data_points:
            model_points:
            threshold:
        Returns: True if it is inlier, False otherwise

        """
        if np.linalg.norm(data_points-model_points)<threshold:
            return True
        else:
            return False
    def ransac_fit(self,data,model,is_inlier,sample_size,iterations,threshold):
        """
            Generic RANSAC implementation

            Args:
            - data: Full dataset
            - model: Function to fit model to a sample
            - is_inlier: Function to check if a point fits the model
            - sample_size: Minimum points needed to fit the model
            - iterations: Number of RANSAC iterations
            - threshold: Inlier determination criterion

            Returns:
            - Best model found
            - Indices of inliers
            """
        best_model = None
        best_inliers = []

        for _ in range(iterations):
            # Randomly sample data
            sample_indices = random.sample(range(len(data)), sample_size)
            sample = [data[i] for i in sample_indices]

            # Fit model to sample
            potential_model = model(sample)

            # Find inliers
            inliers = [
                i for i, point in enumerate(data)
                if is_inlier(potential_model, point, threshold)
            ]

            # Update best model if more inliers found
            if len(inliers) > len(best_inliers):
                best_model = potential_model
                best_inliers = inliers

        return best_model, best_inliers
    def parabola_model(self,params,x,y):
        a,b,c,d,e,f,g,h = params
        z=a*x**2+b*y**2+c*x*y+d*x+e*y+f
        return x,y,z
    def parabola_model_time(self,params,t):
        x_0,y_0,z_0,vx_0,vy_0,vz_0 = params
        x=x_0+vx_0*t
        y=y_0+vy_0*t
        z=z_0+vz_0*t+0.5*self.g*t**2
        return x,y,z
    def direct_fitter(self):
        pass
    def mono_fitter(self,data,odometry=None):
        """
        Fit a parabola with point's position(in meters 2D) in the camera coordinate
        The length of odometry must be the same as the length of the data
        Args:
            data: [[x_c,y_c,t],
                    [x_c,y_c,t]]
            odometry: [[x_w,y_w,z_w,t],] or None
        Returns: Parabola fitted params ndarrays

        """
        # print(len(data))
        # if not len(data)==999:
        #     return None
        if len(data)<self.min_date_sample:
            return None
        t_start = data[0][2]
        rotation_matrix_to=rotate_matrix_coordinate(self.camera_rpy_to[0],self.camera_rpy_to[1],self.camera_rpy_to[2])
        g_vector=np.array([0,0,self.g])
        g_x_c,g_y_c,g_z_c=transform_coordinate(g_vector,rotation_matrix_to,[0,0,0])
        # print(g_x_c,g_y_c,g_z_c)

        A=[]
        B=[]
        for i in range(0,len(data)):
            t=data[i][2]-t_start
            camera_vector = np.concatenate((np.array(data[i][:2]),np.array([1])))
            # print(camera_vector,t)
            # compensate the movement of the robot with odometry data
            # make all points([x_c,y_c]) in camera coordinate relative to the start camera position
            if odometry is not None:
                # Equals to move opposite direction
                translation = -1*np.array(odometry[i][:3])
                # Assume the camera didn't rotate over time
                no_rotation = rotate_matrix_coordinate(0, 0,0)
                camera_vector = transform_coordinate(camera_vector,no_rotation,translation)
                # print(camera_vector)
            # All data points in camera coordinate
            alpha=camera_vector[0]
            beta=camera_vector[1]
            A.append([1, t, 0, 0, -alpha, -alpha * t])
            A.append([0, 0, 1, t, -beta, -beta * t])
            B.append(0.5 * t * t * (g_z_c * alpha - g_x_c))
            B.append(0.5 * t * t * (g_z_c * beta - g_y_c))
        A_array = np.array(A)
        B_array = np.array(B)

        initial_w = [0, 0, 0]
        initial_c = transform_coordinate(initial_w, rotation_matrix_to, [0, 0, 0])
        initial_vw = [0, 0, 0]
        initial_vc = transform_coordinate(initial_vw, rotation_matrix_to, [0, 0, 0])
        initial = np.array([initial_c[0], initial_vc[0], initial_c[1], initial_vc[1], initial_c[2], initial_vc[2]])
        # pseudo_inverse_A = np.linalg.pinv(A_array)
        # solved_parameters = pseudo_inverse_A @ B_array
        #
        # # The params in the camera coordinate
        # least_squares
        # x0_c, vx_c, y0_c, vy_c, z0_c, vz_c = solved_parameters[0][0], solved_parameters[1][0], solved_parameters[2][
        #     0], solved_parameters[3][0], solved_parameters[4][0], solved_parameters[5][0]
        # def residuals(x,A,b):
        #     return A @ x - b

        # solved_parameters = least_squares(residuals, x0, method='trf',args=(A_array,B_array))
        # x0_c, vx_c, y0_c, vy_c, z0_c, vz_c=solved_parameters.x

        def gradient_descent(A, B, initial, learning_rate=0.1, tolerance=0.01, max_iter=100,max_grad_norm=100):

            x = initial
            for i in range(max_iter):
                gradient = -2 * A.T @ (B - A @ x)
                grad_norm = np.linalg.norm(gradient)
                # print(grad_norm)
                if grad_norm > max_grad_norm:
                    gradient = gradient * (max_grad_norm / grad_norm)

                if grad_norm > 50:
                    learning_rate *= 0.5
                elif grad_norm < 10:
                    learning_rate *= 1.05
                x_new = x - learning_rate * gradient
                # print(x_new)
                if np.linalg.norm(x_new - x) < tolerance:
                    # print("stop at iteration %d" % i)
                    break
                x = x_new
            return x
        solved_parameters=gradient_descent(A_array, B_array,initial)
        x0_c, vx_c, y0_c, vy_c, z0_c, vz_c = solved_parameters
        # print(initial)
        # print(x0_c, vx_c, y0_c, vy_c, z0_c, vz_c)
        rotation_matrix_back=rotate_matrix_coordinate(self.camera_rpy_back[0],self.camera_rpy_back[1],self.camera_rpy_back[2])
        x_0,y_0,z_0 = transform_coordinate(np.array([x0_c,y0_c,z0_c]),rotation_matrix_back,[0,0,0])
        vx, vy, vz = transform_coordinate(np.array([vx_c, vy_c, vz_c]), rotation_matrix_back, [0, 0, 0])
        return x_0,y_0,z_0,vx,vy,vz
    def mono_pixel_fitter(self):
        pass




if __name__ == '__main__':
  pass