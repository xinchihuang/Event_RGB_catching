import random
import numpy as np
from scipy.optimize import least_squares
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import transform_coordinate,rotate_matrix_coordinate,pixel_to_world,world_to_pixel
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor
from fitting.ransac_fit import *
class MonoFitter:
    def __init__(self,camera_params,rotation_matrix_back):
        self.camera_params = camera_params
        self.rotation_matrix_back = rotation_matrix_back
    def set_data(self,data):
        self.data = data
    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back)
        initial_guess = [4, 0, 2, -8, 0.1, 1]  # [X0, Y0, Z0, Vx, Vy, Vz]
        timesteps = self.data[:, 3]
        def trajectory_model(params, t):
            X0, Y0, Z0, Vx, Vy, Vz = params
            X_pred = X0 + Vx * t
            Y_pred = Y0 + Vy * t
            Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
            return np.column_stack([X_pred, Y_pred, Z_pred])

        def residuals(params):
            predicted_coords = trajectory_model(params, timesteps)
            return (world_coords[:2] - predicted_coords[:2]).ravel()  # 展平成1D数组

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
class MonoFitter3D:
    def __init__(self,camera_params,rotation_matrix_back,ball_diameter=0.14):
        self.camera_params = camera_params
        self.rotation_matrix_back = rotation_matrix_back
    def set_data(self,data):
        self.data = data
    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back)
        X0, Y0, Z0 = world_coords[0]
        print(X0, Y0, Z0)
        initial_guess = [X0, Y0, Z0, -8, 0, 0]  # [X0, Y0, Z0, Vx, Vy, Vz]
        timesteps = self.data[:, 3]
        print(timesteps)
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
            max_nfev=100,
            xtol=1e-5,
            ftol=1e-5,
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
class MonoFitterRansac:
    def __init__(self, camera_params, rotation_matrix_back):
        self.camera_params = camera_params
        self.rotation_matrix_back = rotation_matrix_back

    def set_data(self, data):
        self.data = data

    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back)
        timesteps = self.data[:, 3]

        # 提取 X 和 Y 方向数据
        X = world_coords[:, 0]
        Y = world_coords[:, 1]
        Z = world_coords[:, 2]

        # 使用 RANSAC 拟合 X, Y, Z
        ransac_X = RANSACRegressor().fit(timesteps.reshape(-1, 1), X)
        ransac_Y = RANSACRegressor().fit(timesteps.reshape(-1, 1), Y)
        ransac_Z = RANSACRegressor().fit(timesteps.reshape(-1, 1), Z)

        # 获取拟合的初始值
        X0, Vx = ransac_X.estimator_.intercept_, ransac_X.estimator_.coef_[0]
        Y0, Vy = ransac_Y.estimator_.intercept_, ransac_Y.estimator_.coef_[0]
        Z0, Vz = ransac_Z.estimator_.intercept_, ransac_Z.estimator_.coef_[0]

        # # 由于 Z 方向受重力影响，我们调整初始 Vz
        # Vz += 0.5 * 9.81 * np.mean(timesteps)  # 估算初始速度

        initial_guess = [X0, Y0, Z0, Vx, Vy, Vz]


        def trajectory_model(params, t):
            X0, Y0, Z0, Vx, Vy, Vz = params
            X_pred = X0 + Vx * t
            Y_pred = Y0 + Vy * t
            Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
            return np.column_stack([X_pred, Y_pred, Z_pred])

        def residuals(params):
            predicted_coords = trajectory_model(params, timesteps)
            return (world_coords[:, :2] - predicted_coords[:, :2]).ravel()

        opt_result = least_squares(
            residuals, initial_guess, max_nfev=200, xtol=1e-6, ftol=1e-6, verbose=0
        )

        success = opt_result.success
        params = opt_result.x.tolist() if success else None
        avg_error = np.mean(np.abs(opt_result.fun)) if success else np.inf

        self.predicted_params = {
            "success": success,
            "points": world_coords,
            "params": {
                "x0": params[0], "y0": params[1], "z0": params[2],
                "vx": params[3], "vy": params[4], "vz": params[5],
            } if success else None,
            "cost": opt_result.cost,
            "reprojection_error": avg_error,
            "iterations": opt_result.nfev,
        }
        return self.predicted_params

    def predict(self):
        params = self.predicted_params["params"]
        x0, y0, z0 = params["x0"], params["y0"], params["z0"]
        vx, vy, vz = params["vx"], params["vy"], params["vz"]

        impact_t = abs(x0 / vx) if vx != 0 else np.inf
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        return impact_point
class MonoPixelFitter:
    def __init__(self,camera_params,rotation_matrix_to,rotation_matrix_back):
        self.camera_params = camera_params
        self.rotation_matrix_to = rotation_matrix_to
        self.rotation_matrix_back = rotation_matrix_back
    def set_data(self,data):
        self.data = data
    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back)
        initial_guess = [5, 0, 2, -8, 0.1, 1]  # [X0, Y0, Z0, Vx, Vy, Vz]
        timesteps = self.data[:, 3]
        uv_obs = self.data[:, :2]
        def trajectory_model(params, t):
            X0, Y0, Z0, Vx, Vy, Vz = params
            X_pred = X0 + Vx * t
            Y_pred = Y0 + Vy * t
            Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
            return np.column_stack([X_pred, Y_pred, Z_pred])

        def residuals(params):
            errors = []
            for t, (u_obs, v_obs) in zip(timesteps, uv_obs):
                # print(t)
                predicted_coords=trajectory_model(params, t)
                # print(predicted_coords)
                u_pred, v_pred,d_pix_pred = world_to_pixel(predicted_coords, self.camera_params, self.rotation_matrix_to)
                if u_pred is None:  # 无效点跳过
                    errors.extend([100000.0, 100000.0])
                else:
                    errors.extend([u_obs - u_pred, v_obs - v_pred])
                # print(errors)
            return np.array(errors).ravel()

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
        t = np.linspace(0, max(timesteps), len(self.data))
        x = params[0] + params[3] * t
        y = params[1] + params[4] * t
        z = params[2] + params[5] * t - 0.5 * 9.81 * t ** 2
        predicted_coord=np.column_stack([x, y, z])
        predeicted_params = {
            'success': success,
            "points": predicted_coord,
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
        params = self.predeicted_params['params']
        x0 = params['x0']
        y0 = params['y0']
        z0 = params['z0']
        vx = params['vx']
        vy = params['vy']
        vz = params['vz']
        print(x0, y0, z0, vx, vy, vz)
        # 碰撞时间：假设球体以 vx 匀速运动到 x=0
        impact_t = abs(x0 / vx)
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        return impact_point
class MonoPixelFitter3D:
    def __init__(self,camera_params,rotation_matrix_to,rotation_matrix_back):
        self.camera_params = camera_params
        self.rotation_matrix_to = rotation_matrix_to
        self.rotation_matrix_back = rotation_matrix_back
    def set_data(self,data):
        self.data = data
    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back)
        X0, Y0, Z0 = world_coords[0]
        initial_guess = [X0, Y0, Z0, -8, 0, 0]  # [X0, Y0, Z0, Vx, Vy, Vz]
        timesteps = self.data[:, 3]
        uv_obs = self.data[:, :3]
        def trajectory_model(params, t):
            X0, Y0, Z0, Vx, Vy, Vz = params
            X_pred = X0 + Vx * t
            Y_pred = Y0 + Vy * t
            Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
            return np.column_stack([X_pred, Y_pred, Z_pred])

        def residuals(params):
            errors = []
            for t, (u_obs, v_obs,d_obs) in zip(timesteps, uv_obs):
                # print(t)
                predicted_coords=trajectory_model(params, t)
                # print(predicted_coords)
                u_pred, v_pred,d_pix_pred = world_to_pixel(predicted_coords, self.camera_params, self.rotation_matrix_to)
                if u_pred is None:  # 无效点跳过
                    errors.extend([100000.0, 100000.0,100000.0])
                else:
                    errors.extend([u_obs - u_pred, v_obs - v_pred,d_pix_pred-d_obs])
                # print(errors)
            return np.array(errors).ravel()

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
        t = np.linspace(0, max(timesteps), len(self.data))
        x = params[0] + params[3] * t
        y = params[1] + params[4] * t
        z = params[2] + params[5] * t - 0.5 * 9.81 * t ** 2
        predicted_coord=np.column_stack([x, y, z])
        predeicted_params = {
            'success': success,
            "points": predicted_coord,
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
        params = self.predeicted_params['params']
        x0 = params['x0']
        y0 = params['y0']
        z0 = params['z0']
        vx = params['vx']
        vy = params['vy']
        vz = params['vz']
        print(x0, y0, z0, vx, vy, vz)
        # 碰撞时间：假设球体以 vx 匀速运动到 x=0
        impact_t = abs(x0 / vx)
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        return impact_point
#not yet finish
class MonoFitter3DRansac:
    def __init__(self, camera_params, rotation_matrix_back):
        self.camera_params = camera_params
        self.rotation_matrix_back = rotation_matrix_back

    def set_data(self, data):
        self.data = data

    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back)
        timesteps = self.data[:, 3]

        # 提取 X 和 Y 方向数据
        X = world_coords[:, 0]
        Y = world_coords[:, 1]
        Z = world_coords[:, 2]

        # 使用 RANSAC 拟合 X, Y, Z
        ransac_X = RANSACRegressor().fit(timesteps.reshape(-1, 1), X)
        ransac_Y = RANSACRegressor().fit(timesteps.reshape(-1, 1), Y)
        ransac_Z = RANSACRegressor().fit(timesteps.reshape(-1, 1), Z)

        # 获取拟合的初始值
        X0, Vx = ransac_X.estimator_.intercept_, ransac_X.estimator_.coef_[0]
        Y0, Vy = ransac_Y.estimator_.intercept_, ransac_Y.estimator_.coef_[0]
        Z0, Vz = ransac_Z.estimator_.intercept_, ransac_Z.estimator_.coef_[0]

        # # 由于 Z 方向受重力影响，我们调整初始 Vz
        # Vz -= 0.5 * 9.81 * np.mean(timesteps)  # 估算初始速度

        initial_guess = [X0, Y0, Z0, Vx, Vy, Vz]
        print(initial_guess)

        def trajectory_model(params, t):
            X0, Y0, Z0, Vx, Vy, Vz = params
            X_pred = X0 + Vx * t
            Y_pred = Y0 + Vy * t
            Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
            return np.column_stack([X_pred, Y_pred, Z_pred])

        def residuals(params):
            predicted_coords = trajectory_model(params, timesteps)
            return (world_coords[:, :] - predicted_coords[:, :]).ravel()

        opt_result = least_squares(
            residuals, initial_guess, max_nfev=200, xtol=1e-6, ftol=1e-6, verbose=0
        )

        success = opt_result.success
        params = opt_result.x.tolist() if success else None
        avg_error = np.mean(np.abs(opt_result.fun)) if success else np.inf

        self.predicted_params = {
            "success": success,
            "points": world_coords,
            "params": {
                "x0": params[0], "y0": params[1], "z0": params[2],
                "vx": params[3], "vy": params[4], "vz": params[5],
            } if success else None,
            "cost": opt_result.cost,
            "reprojection_error": avg_error,
            "iterations": opt_result.nfev,
        }
        return self.predicted_params

    def predict(self):
        params = self.predicted_params["params"]
        x0, y0, z0 = params["x0"], params["y0"], params["z0"]
        vx, vy, vz = params["vx"], params["vy"], params["vz"]

        impact_t = abs(x0 / vx) if vx != 0 else np.inf
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        return impact_point





class MonoEKF:
    def __init__(self, camera_params, rotation_matrix_to, rotation_matrix_back):
        self.camera_params = camera_params
        self.rotation_matrix_to = rotation_matrix_to
        self.rotation_matrix_back = rotation_matrix_back
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
        initial_z=6

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
            camera_velocity = np.array([vx_c, vy_c, 1.4 * vy_c])
            world_velocity = R_back @ camera_velocity - T
            return world_point, world_velocity
        def compute_jacobian(x, R, T=np.zeros(3)):
            x_w, y_w, z_w = x[0, 0], x[1, 0], x[2, 0]
            world_point = np.array([x_w, y_w, z_w])
            cam_point = R @ world_point + T
            x_c, y_c, z_c = cam_point[0], cam_point[1], cam_point[2]
            H = np.zeros((2, 6))
            H[0, 0] = (fx / z_c) * R[0, 0] - (fx * x_c / z_c ** 2) * R[2, 0]
            H[0, 1] = (fx / z_c) * R[0, 1] - (fx * x_c / z_c ** 2) * R[2, 1]
            H[0, 2] = (fx / z_c) * R[0, 2] - (fx * x_c / z_c ** 2) * R[2, 2]
            H[1, 0] = (fy / z_c) * R[1, 0] - (fy * y_c / z_c ** 2) * R[2, 0]
            H[1, 1] = (fy / z_c) * R[1, 1] - (fy * y_c / z_c ** 2) * R[2, 1]
            H[1, 2] = (fy / z_c) * R[1, 2] - (fy * y_c / z_c ** 2) * R[2, 2]
            return H
        ball_memory=self.data
        estimated_trajectory = []
        initial_dt = ball_memory[1][3] - ball_memory[0][3]
        initial_u, initial_v = ball_memory[1][0], ball_memory[1][1]
        initial_vu = (ball_memory[1][0] - ball_memory[0][0]) / initial_dt
        initial_vv = (ball_memory[1][1] - ball_memory[0][1]) / initial_dt
        world_point, world_velocity = estimate_state(initial_u, initial_v, initial_z, initial_vu, initial_vv,
                                                     self.rotation_matrix_back)
        x = np.array([world_point[0], world_point[1], world_point[2], world_velocity[0], world_velocity[1],
                      world_velocity[2]]).reshape(-1, 1)
        for step in range(1, ball_memory.shape[0]):
            dt = ball_memory[step][3] - ball_memory[step - 1][3]
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
            H = compute_jacobian(x, self.rotation_matrix_to)
            z_pred = h_rotation(x, self.rotation_matrix_to)
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
            # print(u, v, z_c, vu, vv, vz)
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy
            vx_c = vu * z_c / fx
            vy_c = vv * z_c / fy
            vz_c = vz * z_c / fx
            # print(vx_c,vy_c,vz_c)
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

class MonoRansac:
    def __init__(self,camera_params,rotation_matrix_to,rotation_matrix_back,ball_diameter=0.104):
        self.camera_params = camera_params
        self.rotation_matrix_to = rotation_matrix_to
        self.rotation_matrix_back = rotation_matrix_back
        self.ball_diameter = ball_diameter

    def set_data(self,data):
        self.data = data
    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back, self.ball_diameter)

        init_position, velocity, inliers, points_3d = adaptive_ransac_parabola_3d(world_coords, self.data[:, 3],)
        predeicted_params = {
            "points": points_3d,
            'params': {
                'x0': init_position[0],
                'y0': init_position[1],
                'z0': init_position[2],
                'vx': velocity[0],
                'vy': velocity[1],
                'vz': velocity[2],
            },
            'inliers': inliers,
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
        print(x0, y0, z0, vx, vy, vz)
        impact_t = abs(x0 / vx)
        impact_y = y0 + vy * impact_t
        impact_z = z0 + vz * impact_t - 0.5 * 9.81 * impact_t ** 2
        impact_point = [0, impact_y, impact_z]
        print(impact_point)
        return impact_point

class MonoRansacDrag:
    def __init__(self, camera_params, rotation_matrix_to, rotation_matrix_back, ball_diameter=0.104,
                 drag_k_init=0.05, drag_k_bounds=(1e-4, 5.0)):
        self.camera_params = camera_params
        self.rotation_matrix_to = rotation_matrix_to
        self.rotation_matrix_back = rotation_matrix_back
        self.ball_diameter = ball_diameter
        self.drag_k_init = drag_k_init
        self.drag_k_bounds = drag_k_bounds

    def set_data(self, data):
        self.data = data

    def _trajectory_model(self, params, t):
        x0, y0, z0, vx, vy, vz, k = params
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        p0 = np.array([x0, y0, z0], dtype=float).reshape(1, -1)
        v0 = np.array([vx, vy, vz], dtype=float).reshape(1, -1)
        g = GRAVITY.reshape(1, -1)
        if k < 1e-6:
            return p0 + v0 * t + 0.5 * g * t ** 2
        exp_term = np.exp(-k * t)
        drag_term = (v0 - g / k) * (1.0 - exp_term) / k + g * t / k
        return p0 + drag_term

    def _impact_time(self, x0, vx, k):
        if vx == 0:
            return np.inf
        if k < 1e-6:
            return abs(x0 / vx)
        term = 1.0 + k * x0 / vx
        if term <= 0:
            return abs(x0 / vx)
        impact_t = -np.log(term) / k
        if impact_t < 0:
            return abs(x0 / vx)
        return impact_t

    def get_params(self):
        world_coords = pixel_to_world(self.data, self.camera_params, self.rotation_matrix_back, self.ball_diameter)
        timesteps = self.data[:, 3]

        init_position, velocity, inliers, points_3d = adaptive_ransac_parabola_3d(
            world_coords,
            timesteps,
            use_drag=True,
            drag_k=self.drag_k_init,
        )
        fit_points = world_coords[inliers] if inliers is not None else world_coords
        fit_times = timesteps[inliers] if inliers is not None else timesteps

        initial_guess = [
            init_position[0], init_position[1], init_position[2],
            velocity[0], velocity[1], velocity[2],
            self.drag_k_init,
        ]
        lower = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, self.drag_k_bounds[0]]
        upper = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, self.drag_k_bounds[1]]

        def residuals(params):
            predicted_coords = self._trajectory_model(params, fit_times)
            return (fit_points - predicted_coords).ravel()

        opt_result = least_squares(
            residuals,
            initial_guess,
            bounds=(lower, upper),
            max_nfev=200,
            xtol=1e-6,
            ftol=1e-6,
            verbose=0
        )
        params = opt_result.x.tolist() if opt_result.success else initial_guess
        predeicted_params = {
            "points": points_3d,
            "params": {
                "x0": params[0],
                "y0": params[1],
                "z0": params[2],
                "vx": params[3],
                "vy": params[4],
                "vz": params[5],
                "k": params[6],
            },
            "inliers": inliers,
        }
        self.predeicted_params = predeicted_params
        return predeicted_params

    def predict(self):
        params = self.predeicted_params["params"]
        x0 = params["x0"]
        y0 = params["y0"]
        z0 = params["z0"]
        vx = params["vx"]
        vy = params["vy"]
        vz = params["vz"]
        k = params.get("k", self.drag_k_init)

        impact_t = self._impact_time(x0, vx, k)
        impact_point = self._trajectory_model([x0, y0, z0, vx, vy, vz, k], [impact_t])[0]
        impact_point[0] = 0
        return impact_point.tolist()




if __name__ == '__main__':
  pass
