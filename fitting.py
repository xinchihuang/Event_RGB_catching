import numpy as np
from scipy.optimize import least_squares

import numpy as np
from scipy.optimize import least_squares
from utils import pixel_to_world


def fit_3d_trajectory_with_distortion_old(observations, camera_params, rotation_matrix, initial_guess, max_iter=200):
    """
    考虑镜头畸变的3D抛体运动轨迹拟合

    参数：
    observations : list of tuples - (u像素, v像素, 时间t)
    camera_params : dict - 必须包含:
        'fx', 'fy' : 焦距
        'cx', 'cy' : 主点
        'dist_coeffs' : 畸变系数 [k1, k2, p1, p2, k3]
    rotation_matrix : np.array - 3x3旋转矩阵
    initial_guess : list - 初始猜测参数 [X0, Y0, Z0, Vx, Vy, Vz, g]
    max_iter : int - 最大迭代次数

    返回：
    {
        'success': bool,
        'params': 优化后的运动参数,
        'cost': float,
        'reprojection_error': float,
        'iterations': int
    }
    """
    # 预处理观测数据
    obs_array = np.array(observations)
    timesteps = obs_array[:, 2]
    uv_obs = obs_array[:, :2]
    # print(obs_array)
    # 提取畸变系数 (兼容无畸变情况)
    dist = camera_params.get('dist_coeffs', np.zeros(5))
    k1, k2, p1, p2, k3 = dist if len(dist) == 5 else np.append(dist, [0] * (5 - len(dist)))

    def apply_distortion(x, y):
        """应用Brown-Conrady畸变模型"""
        r2 = x ** 2 + y ** 2
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_dist = y * radial + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
        return x_dist, y_dist

    def world_to_pixel(params, t):
        """完整投影流程：世界坐标系->相机坐标系->像素坐标"""
        # 计算世界坐标系中的3D位置
        X0, Y0, Z0, Vx, Vy, Vz= params
        Xw = X0 + Vx * t
        Yw = Y0 + Vy * t
        Zw = Z0 + Vz * t - 0.5 * 9.81 * t ** 2

        # 转换到相机坐标系
        Xc, Yc, Zc = rotation_matrix @ np.array([Xw, Yw, Zw])

        # 避免除零错误
        if Zw < 1. or Zw>2.2 or Xw<0.4 or Xw>4 or Yw<-0.5 or Yw>0.5:
            return None, None

        # 归一化坐标
        x = Xc / Zc
        y = Yc / Zc

        # 应用畸变
        x_dist, y_dist = apply_distortion(x, y)

        # 转换为像素坐标
        u = camera_params['fx'] * x_dist + camera_params['cx']
        v = camera_params['fy'] * y_dist + camera_params['cy']
        # print(u,v)
        return u, v

    def residual(params):
        errors = []
        for t, (u_obs, v_obs) in zip(timesteps, uv_obs):
            # print(t)
            u_pred, v_pred = world_to_pixel(params, t)
            if u_pred is None:  # 无效点跳过
                errors.extend([100000.0, 100000.0])
            else:
                errors.extend([u_obs - u_pred, v_obs - v_pred])
            # print(errors)
        return np.array(errors)

    # 执行优化（增加参数缩放）
    opt_result = least_squares(
        residual,
        initial_guess,
        # bounds=(0,10),
        # method='trf',
        max_nfev=max_iter,
        xtol=1e-6,
        ftol=1e-6,
        verbose=0
    )

    # 结果整理
    success = opt_result.success
    params = opt_result.x.tolist() if success else None
    avg_error = np.mean(np.abs(opt_result.fun)) if success else np.inf

    return {
        'success': success,
        'params': {
            'x0': params[0],
            'y0': params[1],
            'z0': params[2],
            'vx': params[3],
            'vy': params[4],
            'vz': params[5],
        },
        'cost': opt_result.cost,
        'reprojection_error': avg_error,
        'iterations': opt_result.nfev
    }


def fit_3d_trajectory_with_distortion(observations, camera_params, rotation_matrix):
    # print(observations)
    world_coords = pixel_to_world(observations, camera_params, rotation_matrix)
    # print(world_coords)
    X0, Y0, Z0 = world_coords[0]
    initial_guess = [X0, Y0, Z0, 0, 0, 0]  # [X0, Y0, Z0, Vx, Vy, Vz]
    timesteps=observations[:,3]
    print(timesteps)
    def trajectory_model(params, t):
        """计算给定参数下的预测轨迹"""
        X0, Y0, Z0, Vx, Vy, Vz = params
        X_pred = X0 + Vx * t
        Y_pred = Y0 + Vy * t
        Z_pred = Z0 + Vz * t - 0.5 * 9.81 * t ** 2
        return np.column_stack([X_pred, Y_pred, Z_pred])

    def residuals(params):
        """计算观测数据和预测轨迹之间的误差"""
        predicted_coords = trajectory_model(params, timesteps)
        return (world_coords - predicted_coords).ravel()  # 展平成1D数组

    # 使用非线性最小二乘法优化
    opt_result = least_squares(
        residuals,
        initial_guess,
        max_nfev=200,
        xtol=1e-6,
        ftol=1e-6,
        verbose=0
    )

    # 解析优化结果
    success = opt_result.success
    params = opt_result.x.tolist() if success else None
    avg_error = np.mean(np.abs(opt_result.fun)) if success else np.inf

    return {
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

# 示例调用
if __name__ == "__main__":
    # 定义输入参数
    camera_config= {
        'fx': 800,
        'fy': 800,
        'cx': 320,
        'cy': 240,
        'dist_coeffs': [-0.2, 0.1, 0.001, -0.002, 0.05]  # k1, k2, p1, p2, k3
    }

    # 单位矩阵表示世界坐标系与相机坐标系对齐
    rotation_mat = np.eye(3)

    # 初始猜测参数（需要根据实际情况调整）
    init_guess = [0.5, 1.0, 4.5, 1.3, 3.5, 0.3]
    ball_memory = [
        [100, 100, 0.1],
        [110, 110, 0.2],
        [120, 120, 0.3],
        [130, 130, 0.4],
        [140, 140, 0.5],
        [150, 150, 0.6],
        [160, 160, 0.7],
    ]

    # 执行拟合
    result = fit_3d_trajectory_with_distortion(
        observations=ball_memory,
        camera_params=camera_config,
        rotation_matrix=rotation_mat,
        initial_guess=init_guess
    )

    # 输出结果
    if result['success']:
        print("拟合成功的运动参数：")
        for k, v in result['params'].items():
            print(f"{k}: {v:.3f}")
    else:
        print(f"拟合失败")