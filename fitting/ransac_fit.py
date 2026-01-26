import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import BboxKeys, compute_area_from_bboxes
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

GRAVITY = np.array([0, 0, -9.81])  # in positive y direction


def _drag_alpha_beta(time, k):
    t = np.asarray(time, dtype=float)
    if k < 1e-6:
        alpha = t
        beta = 0.5 * t ** 2
    else:
        exp_term = np.exp(-k * t)
        alpha = (1.0 - exp_term) / k
        beta = t / k - (1.0 - exp_term) / (k ** 2)
    return alpha, beta


def _trajectory_drag(p, v, time, k):
    alpha, beta = _drag_alpha_beta(time, k)
    return (p.reshape(1, -1)
            + v.reshape(1, -1) * alpha.reshape(-1, 1)
            + GRAVITY.reshape(1, -1) * beta.reshape(-1, 1))

def bboxes_to_3d_points(bboxes, area_ball=.1 ** 2, camera_matrix=None, distortion_coeffs=None, distortion_model=None):
    # get depth from area
    area = compute_area_from_bboxes(bboxes) * np.pi / 4
    area_ratio = (area_ball / area).astype('float')
    depth = np.sqrt(area_ratio) * camera_matrix[0, 0]

    # compute 3D points from object center points
    xy = bboxes[:, [BboxKeys.center_x, BboxKeys.center_y]].astype('float')
    xy_norm = cv2.undistortPoints(xy, camera_matrix, distortion_coeffs)  # return normalized coords
    xy_norm = xy_norm[:, 0]

    xy_norm = np.concatenate([xy_norm, np.ones_like(xy[:, :1])], axis=-1)
    points = xy_norm * depth.reshape((-1, 1))

    return points


def minimal_solution_parabola(points_3d, time):
    t0, t1 = time
    p0, p1 = points_3d

    p0_gravity_comp = p0 - GRAVITY * .5 * t0 ** 2
    p1_gravity_comp = p1 - GRAVITY * .5 * t1 ** 2

    v = (p1_gravity_comp - p0_gravity_comp) / (t1 - t0)
    p = p0_gravity_comp - v * t0
    # print(p1_gravity_comp,p0_gravity_comp,t1,t0)

    return p, v


def minimal_solution_drag(points_3d, time, k):
    time = np.asarray(time, dtype=float)
    if k < 1e-6:
        return minimal_solution_parabola(points_3d, time)
    alpha, beta = _drag_alpha_beta(time, k)
    A = np.array([[1.0, alpha[0]], [1.0, alpha[1]]], dtype=float)
    if np.isclose(np.linalg.det(A), 0.0):
        return minimal_solution_parabola(points_3d, time)
    adjusted = points_3d - GRAVITY.reshape(1, -1) * beta.reshape(-1, 1)
    p = np.zeros(3)
    v = np.zeros(3)
    for axis in range(3):
        sol = np.linalg.solve(A, adjusted[:, axis])
        p[axis] = sol[0]
        v[axis] = sol[1]
    return p, v


def ransac_iter_from_confidence_and_outlier_probability(outlier_probability, confidence=0.99):
    return int(np.log(1 - confidence) / np.log(1 - (1 - outlier_probability) ** 2))


def least_squares_solution_parabola(points_3d, time):
    # solves a least squares problem
    # error = sum (bx - vx t - px)^2 + (by - 1/2 g t^2 - vy t - py)^2 + (bz - vz t - pz)^2
    # the solution can be separated over vx, px, vy, py and vz, pz
    # 0 = sum bx   - vx sum t   - px n
    # 0 = sum t bx - vx sum t^2 - px sum t
    points_3d_gravity_comp = points_3d - 0.5 * GRAVITY.reshape((1, -1)) * time.reshape((-1, 1)) ** 2

    n = len(points_3d)
    tp_sum = (points_3d_gravity_comp * time.reshape((-1, 1))).sum(0)
    p_sum = points_3d_gravity_comp.sum(0)
    t_sum = time.sum()
    # print(t_sum)
    t_2_sum = (time ** 2).sum()
    p_opt = (t_sum * tp_sum - p_sum * t_2_sum) / (t_sum ** 2 - n * t_2_sum)
    v_opt = (t_sum * p_sum - n * tp_sum) / (t_sum ** 2 - n * t_2_sum)

    return p_opt, v_opt


def least_squares_solution_drag(points_3d, time, k):
    time = np.asarray(time, dtype=float)
    if k < 1e-6:
        return least_squares_solution_parabola(points_3d, time)
    alpha, beta = _drag_alpha_beta(time, k)
    adjusted = points_3d - GRAVITY.reshape(1, -1) * beta.reshape(-1, 1)
    n = len(points_3d)
    alpha_sum = alpha.sum()
    alpha_2_sum = (alpha ** 2).sum()
    adjusted_sum = adjusted.sum(0)
    alpha_adjusted_sum = (adjusted * alpha.reshape(-1, 1)).sum(0)
    denom = alpha_sum ** 2 - n * alpha_2_sum
    if np.isclose(denom, 0.0):
        return least_squares_solution_parabola(points_3d, time)
    p_opt = (alpha_sum * alpha_adjusted_sum - adjusted_sum * alpha_2_sum) / denom
    v_opt = (alpha_sum * adjusted_sum - n * alpha_adjusted_sum) / denom
    return p_opt, v_opt


def compute_inliers(points_3d, time, p, v, inlier_threshold=0.1):
    points_3d_gravity_comp = points_3d - 0.5 * GRAVITY.reshape((1, -1)) * time.reshape((-1, 1)) ** 2
    errors = (points_3d_gravity_comp - v.reshape((1, -1)) * time.reshape((-1, 1)) - p.reshape((1, -1)))
    inliers = (errors ** 2).sum(-1) < inlier_threshold ** 2
    return inliers


def compute_inliers_drag(points_3d, time, p, v, k, inlier_threshold=0.1):
    predicted = _trajectory_drag(p, v, time, k)
    errors = points_3d - predicted
    inliers = (errors ** 2).sum(-1) < inlier_threshold ** 2
    return inliers


def reasonable_estimate(p, v):
    # return True
    # impact_point = compute_impact_point(p, v)
    # if v[2] > 0:
    #     return False
    # xlims = [-2, 2]
    # ylims = [0, 2]
    return True  # (impact_point[0] <= xlims[1] and impact_point[0] >= xlims[0] and impact_point[1] <= ylims[1] and impact_point[1] >= ylims[0])


def adaptive_ransac_parabola_3d(points_3d, time, confidence=0.9, num_max_ransac_iter=500,
                                inlier_threshold=0.35, use_drag=False, drag_k=0.05):
    # parabola has equation p(t) = 1/2 g t^2 e_y + v t + p
    # unknowns are p and v
    num_ransac_iter = num_max_ransac_iter
    num_iter_done = 0

    max_num_inliers = 0
    max_p = None
    max_v = None
    # print(time)
    while num_iter_done < num_ransac_iter:
        indices = np.random.choice(len(points_3d), size=(2,), replace=False)
        if use_drag:
            p, v = minimal_solution_drag(points_3d[indices], time[indices], drag_k)
        else:
            p, v = minimal_solution_parabola(points_3d[indices], time[indices])


        if reasonable_estimate(p, v):
            if use_drag:
                inliers = compute_inliers_drag(points_3d, time, p, v, drag_k, inlier_threshold=inlier_threshold)
            else:
                inliers = compute_inliers(points_3d, time, p, v, inlier_threshold=inlier_threshold)
        else:
            inliers = np.full_like(points_3d[:, 0], fill_value=False, dtype="bool")
        num_inliers = inliers.sum()

        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            max_p = p
            max_v = v

        # estimated number of outliers and estimate num iterations
        estimated_oulier_prob = 1 - num_inliers / len(points_3d)
        # threshold the estimated outlier probability
        if estimated_oulier_prob < 0.000001:
            estimated_oulier_prob = 0.000001
        elif estimated_oulier_prob > 0.99999:
            estimated_oulier_prob = 0.99999
        num_ransac_iter_computed = ransac_iter_from_confidence_and_outlier_probability(estimated_oulier_prob,
                                                                                       confidence)
        num_ransac_iter = min([num_ransac_iter_computed, num_max_ransac_iter])

        num_iter_done += 1

    if use_drag:
        p_opt, v_opt = least_squares_solution_drag(points_3d[inliers], time[inliers], drag_k)
    else:
        p_opt, v_opt = least_squares_solution_parabola(points_3d[inliers], time[inliers])
    # p_opt, v_opt = max_p, max_v #least_squares_solution_parabola(points_3d[inliers], time[inliers])
    # print(f"NUM OF INLIERS {num_inliers}, NUM OF POINTS {points_3d.shape[0]}")
    # print("INLIERS")
    # print(points_3d[inliers])
    # print("FULL ARRAY")
    # print(points_3d)
    # print(time[inliers])
    return p_opt, v_opt, inliers, points_3d


def project_3d_to_camera(points_3d, camera_matrix, distortion_coeffs, distortion_model):
    pixels = np.einsum("ij,nj->ni", camera_matrix, points_3d)
    pixels = pixels[:, :2] / pixels[:, -1:]
    map_xy = np.stack(
        cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, camera_matrix, (640, 480), cv2.CV_32FC1),
        axis=-1)
    x, y = pixels.astype("int32").T
    mask = (x >= 0) & (y >= 0) & (x < 640) & (y < 480)
    pixels_dist = map_xy[y[mask], x[mask]]
    return pixels_dist


def fit_parabola_to_bboxes(points_4d, confidence=0.99, num_max_ransac_iter=100, inlier_threshold=0.1, debug=False,
                           ax=None, handles=None):
    points_4d[:, -1] -= points_4d[0, -1]

    points_3d = points_4d[:, :3]
    # print(points_3d,points_4d[:,-1])
    init_position, velocity, inliers, points_3d = adaptive_ransac_parabola_3d(points_3d, points_4d[:, -1], confidence,
                                                                              num_max_ransac_iter, inlier_threshold)
    return init_position, velocity, inliers, points_3d


if __name__ == "__main__":
    adaptive_ransac_parabola_3d()
