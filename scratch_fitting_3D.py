import numpy as np
from fitting import fit_3d_trajectory_with_distortion
from utils import rotate_matrix_coordinate
import os
from plot.plot_3D import draw_3D_trajetory,draw_multi_3D_trajetory
from parabola_fitter import MonoPixelFitter, MonoFitter, MonoFitter3D,MonoPixelFitter3D,MonoRansac


def cal_position(point_index,y_bias=0,z_bias=0.8):
    row=point_index//10
    col=point_index%10
    y=(row)*0.1+y_bias+0.05
    z=(col)*0.1+z_bias+0.05
    return [0,y,z]
def compute_distance_stats(array1, array2):
    if array1.ndim == 1:
        array1 = array1.reshape(1, -1)
    if array2.ndim == 1:
        array2 = array2.reshape(1, -1)

    diff = array1[:, np.newaxis, :] - array2[np.newaxis, :, :]
    squared_diff = diff ** 2
    sum_squared_diff = np.sum(squared_diff, axis=2)
    distances = np.sqrt(sum_squared_diff)

    mean = np.mean(distances)
    variance = np.var(distances)

    return mean, variance


def main():
    """ Main """

    plot_index=10
    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels_3D"
    camera_config = {
        'fx': 739.9,
        'fy': 739.9,
        'cx': 360,
        'cy': 720,
        'dist_'
        'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]

    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])


    label = [14,-63,-62,54,-74,53,22,-54,44,-61]
    # fitter = MonoFitter3D(camera_params=camera_config,rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter(camera_params=camera_config, rotation_matrix_to=rotation_matrix_to,
    #                          rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter3DRansac(camera_params=camera_config, rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter(camera_params=camera_config, rotation_matrix_back=rotation_matrix_back)
    # fitter = PixelSizeFitter(camera_params=camera_config,rotation_matrix_back=rotation_matrix_back)
    fitter = MonoRansac(camera_params=camera_config,rotation_matrix_to=rotation_matrix_to,rotation_matrix_back=rotation_matrix_back)
    exp_results = []
    for i in range(0,plot_index):
        ball_memory=np.load(f"{save_dir}/{i}.npy")
        ball_memory=ball_memory[:15]
        if len(ball_memory)==0:
            continue

        observations=ball_memory
        observations[:,3]=observations[:,3]-observations[0,3]
        fitter.set_data(observations)
        result=fitter.get_params()

        impact_point_real=cal_position(label[i])
        ball = result['points']

        impact_point=fitter.predict()


        dict_to_save={'points' : ball,
             'impact':impact_point,
             'label':impact_point_real,
             }
        exp_results.append(dict_to_save)
    draw_multi_3D_trajetory(raw_data=exp_results)

if __name__ == "__main__":
    main()