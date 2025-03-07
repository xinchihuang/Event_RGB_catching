import numpy as np
# rotation
def rotate_matrix_coordinate(r_x, r_y, r_z):
    """
    Create right hand coordinate rotation matrix for coordinate rotation
    Note this is for coordinate rotation only

    Args:
        r_x: rotation angle(degree) in x axis(counter-clockwise direction from zero) y->z
        r_y: rotation angle in y axis x->z
        r_z: rotation angle in z axis x->y

    Returns: rotation matrix in x-y-z order relative to the rotation axis

    """
    r_x = np.radians(-r_x)
    r_y = np.radians(-r_y)
    r_z = np.radians(-r_z)
    Rotate_x = np.array([
        [1, 0, 0],
        [0, np.cos(r_x), -np.sin(r_x)],
        [0, np.sin(r_x), np.cos(r_x)]
    ])
    Rotate_y = np.array([
        [np.cos(r_y), 0, np.sin(r_y)],
        [0, 1, 0],
        [-np.sin(r_y), 0, np.cos(r_y)]
    ])
    Rotate_z = np.array([
        [np.cos(r_z), -np.sin(r_z), 0],
        [np.sin(r_z), np.cos(r_z), 0],
        [0, 0, 1]
    ])
    return Rotate_z @ Rotate_y @ Rotate_x

# undistort
def batch_undistort_pixel(u, v, fx, fy, cx, cy, k1, k2, p1, p2, k3, max_iter=5):
    """批量去畸变：像素坐标 -> 归一化坐标"""
    x = (u - cx) / fx
    y = (v - cy) / fy

    for _ in range(max_iter):
        r2 = x ** 2 + y ** 2
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x = (x - 2 * p1 * x * y - p2 * (r2 + 2 * x ** 2)) / radial
        y = (y - 2 * p2 * x * y - p1 * (r2 + 2 * y ** 2)) / radial

    return x, y
# pixel_to_world
def pixel_to_world(observations, camera_params, rotation_matrix, ball_diameter=0.14):
    """
    批量计算像素坐标转换为世界坐标

    参数：
    observations : ndarray (N, 4) - 每行 (u, v, d_pix, t)
    camera_params : dict - 相机参数
    rotation_matrix_back : ndarray (3,3) - 相机到世界坐标旋转矩阵
    ball_diameter : float - 球直径（米）

    返回：
    world_coords : ndarray (N, 3) - 世界坐标 (Xw, Yw, Zw)
    """

    # 提取像素坐标 (u, v) 和像素直径 d_pix
    print(observations)
    u = observations[:, 0]
    v = observations[:, 1]
    d_pix = observations[:, 2]

    # 相机参数
    fx, fy = camera_params['fx'], camera_params['fy']
    cx, cy = camera_params['cx'], camera_params['cy']
    dist_coeffs = camera_params.get('dist_coeffs', np.zeros(5))
    k1, k2, p1, p2, k3 = dist_coeffs if len(dist_coeffs) == 5 else np.append(dist_coeffs,
                                                                             [0] * (5 - len(dist_coeffs)))
    x_norm, y_norm = batch_undistort_pixel(u, v, fx, fy, cx, cy, k1, k2, p1, p2, k3)
    Zc = (fx * ball_diameter) / d_pix
    Xc = x_norm
    Yc = y_norm
    camera_coords = np.stack([Xc, Yc, Zc], axis=1)  # (N, 3)
    world_coords = (rotation_matrix @ camera_coords.T).T  # (N,3)
    return camera_coords
