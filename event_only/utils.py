import cv2
import numpy as np
import enum
# from AMTTrack.tracking.scratch import image


# coordinate transformation

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
def translation_vector_coordinate(t_x,t_y,t_z):
    return np.array((t_x,t_y,t_z))
def transform_coordinate(points,rotation_matrix,translation_vector):
    """
    3D transformation of a coordinate
    Get the point's new coordinate
    Rotation
    Args:
        points: Old coordinate
        rotation_matrix: Coordinate rotation matrix
        translation_vector:

    Returns: New coordinate of the point

    """
    # calculate the translation first
    points_translated=np.array(points)-translation_vector
    new_points=rotation_matrix@points_translated
    return new_points

# distortion


def distort(u, v, fx, fy, cx, cy, k1, k2, p1, p2, k3):
    """
    Applies lens distortion to the given 2D points (u, v) using camera intrinsic parameters and distortion coefficients.

    Parameters:
    u, v     - Arrays of 2D undistorted coordinates.
    fx, fy   - Focal lengths of the camera.
    cx, cy   - Principal point (optical center) of the camera.
    k1, k2, k3 - Radial distortion coefficients.
    p1, p2   - Tangential distortion coefficients.

    Returns:
    new_u, new_v - Arrays of 2D distorted coordinates in **Pixels**.
    """

    # Construct the camera intrinsic matrix
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)

    # Construct the distortion coefficient array
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Set rotation and translation vectors to zero (assumes no transformation)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    # Convert 2D points (u, v) into homogeneous 3D points (x, y, 1) in normalized image space
    undistorted_points = np.stack((u, v, np.ones(len(u))), axis=-1)

    # Apply distortion using OpenCV's projectPoints function
    image_points, _ = cv2.projectPoints(undistorted_points, rvec, tvec, cameraMatrix, distCoeffs)

    # Extract the distorted coordinates
    new_u = image_points[:, 0, 0]
    new_v = image_points[:, 0, 1]

    return new_u, new_v
def undistort(u, v, fx, fy, cx, cy, k1, k2, p1, p2, k3):
    """
    Removes lens distortion from the given 2D points (u, v) using camera intrinsic parameters and distortion coefficients.

    Parameters:
    u, v     - Arrays of 2D distorted coordinates.
    fx, fy   - Focal lengths of the camera.
    cx, cy   - Principal point (optical center) of the camera.
    k1, k2, k3 - Radial distortion coefficients.
    p1, p2   - Tangential distortion coefficients.

    Returns:
    new_u, new_v - Arrays of **Normalized (z=1)** 2D undistorted coordinates.
    """

    # Construct the camera intrinsic matrix
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)

    # Construct the distortion coefficient array
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Stack u, v into an (N,2) array for processing
    points = np.stack((u, v), axis=-1)

    # Apply undistortion using OpenCV's undistortPoints function
    undistort_points = cv2.undistortPoints(points, cameraMatrix, distCoeffs, None, None)

    # Extract the undistorted coordinates
    new_u = undistort_points[:, 0, 0]
    new_v = undistort_points[:, 0, 1]

    return new_u, new_v
# pixel_to_world
def pixel_to_world(observations, camera_params, rotation_matrix_back, ball_diameter=0.097):
    """
    Converts pixel coordinates to world coordinates in batch processing.

    Parameters:
    observations : ndarray (N, 4) - Each row contains (u, v, d_pix, t)
    camera_params : dict - Camera intrinsic parameters
    rotation_matrix : ndarray (3,3) - Rotation matrix from camera frame to world frame
    ball_diameter : float - The real-world diameter of the ball (meters), default is 0.14m

    Returns:
    world_coords : ndarray (N, 3) - World coordinates (Xw, Yw, Zw)
    """
    u = observations[:, 0]
    v = observations[:, 1]
    d_pix = observations[:, 2]

    fx, fy = camera_params['fx'], camera_params['fy']
    cx, cy = camera_params['cx'], camera_params['cy']
    dist_coeffs = camera_params.get('dist_coeffs', np.zeros(5))
    k1, k2, p1, p2, k3 = dist_coeffs if len(dist_coeffs) == 5 else np.append(dist_coeffs,
                                                                             [0] * (5 - len(dist_coeffs)))
    # print(u,v)
    x_norm, y_norm = undistort(u, v, fx, fy, cx, cy, k1, k2, p1, p2, k3)
    Zc = (fx * ball_diameter) / d_pix
    Xc = x_norm*Zc
    Yc = y_norm*Zc
    camera_coords = np.stack([Xc, Yc, Zc], axis=1)  # (N, 3)
    world_coords = (rotation_matrix_back @ camera_coords.T).T  # (N,3)

    return world_coords
def world_to_pixel(observations, camera_params, rotation_matrix_to, ball_diameter=0.097):

    x_w = observations[:, 0]
    y_w = observations[:, 1]
    z_w = observations[:, 2]
    if z_w < 1. or z_w > 3 or x_w < 0.4 or x_w > 4 or y_w < -1 or y_w > 1:
        return None, None, None
    world_coords = np.stack([x_w,y_w,z_w], axis=1)  # (N, 3)
    camera_coords = (rotation_matrix_to @ world_coords.T).T  # (N,3)
    x_c= camera_coords[:,0]
    y_c= camera_coords[:,1]
    z_c= camera_coords[:,2]
    fx, fy = camera_params['fx'], camera_params['fy']
    cx, cy = camera_params['cx'], camera_params['cy']
    dist_coeffs = camera_params.get('dist_coeffs', np.zeros(5))
    k1, k2, p1, p2, k3 = dist_coeffs if len(dist_coeffs) == 5 else np.append(dist_coeffs,
                                                                             [0] * (5 - len(dist_coeffs)))
    u, v = distort(x_c/z_c, y_c/z_c, fx, fy, cx, cy, k1, k2, p1, p2, k3)

    d_pix=fx*ball_diameter/z_c

    return u,v,d_pix
####
class BboxKeys(enum.IntEnum):
    time = 0
    width = 1
    height = 2
    center_x = 3
    center_y = 4
    angle = 5
    image_idx = 6

def load_bboxes(path):
    data = np.genfromtxt(path, dtype="float64")
    data[:,BboxKeys.time] -= data[0,BboxKeys.time]
    data[:,BboxKeys.time] /= 1e9
    return data

def filter_to_max_bboxes(bboxes):
    image_idx_unique = np.unique(bboxes[:, BboxKeys.image_idx])

    max_bboxes = []
    for idx in image_idx_unique:
        sub_data = bboxes[bboxes[:, BboxKeys.image_idx] == idx]
        argmax = compute_area_from_bboxes(sub_data).argmax(-1)
        max_bbox = sub_data[argmax]
        max_bboxes.append(max_bbox)
    bboxes = np.stack(max_bboxes)

    return bboxes

def filter_bboxes_to_interval(bboxes, t0, t1):
    bboxes = bboxes[(t0 < bboxes[:,BboxKeys.time]) & (t1 > bboxes[:,BboxKeys.time])]
    bboxes[:,BboxKeys.time] -= bboxes[0,BboxKeys.time]
    return bboxes

def compute_area_from_bboxes(bboxes):
    return bboxes[:, [BboxKeys.width, BboxKeys.height]].prod(-1)