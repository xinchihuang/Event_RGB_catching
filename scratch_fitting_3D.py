import numpy as np
from fitting import fit_3d_trajectory_with_distortion
from utils import rotate_matrix_coordinate
import os
from plot.plot_3D import draw_3D_trajetory,draw_multi_3D_trajetory,draw_multi_3D_trajetory_cls
from parabola_fitter import MonoPixelFitter, MonoFitter, MonoFitter3D,MonoPixelFitter3D,MonoRansac


def cal_position(point_index,y_bias=0,z_bias=0.8):
    if point_index>0:
        row=point_index//10
        col=point_index%10
        y=(col)*0.1+y_bias+0.05
        z=(row)*0.1+z_bias+0.05
        return [0,y,z]
    else:
        point_index=-point_index
        row = point_index // 10
        col = point_index % 10
        # print(col,row)
        y = -(col) * 0.1 + y_bias + 0.05
        z = (row) * 0.1 + z_bias + 0.05
        return [0, y, z]


def main():
    """ Main """

    plot_index=50
    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels_3D"
    camera_config = {
        'fx': 739.9,
        'fy': 739.9,
        'cx': 310,
        'cy': 710,
        'dist_'
        'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]

    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])


    # label = [14,-44,-62,54,-74,63,22,-54,44,-61,24,62,-53,72,32,33,-35,43,51,-24]
    label = [-61,31,-61,0,0,-11,-21,-42,-51,73,
             -31,-13,-51,52,41,-51,-41,61,-31,62,
             -41,12,52,-61,-14,-13,72,46,53,-55,
             21,33,24,21,41,21,-42,-42,61,61,
             62,43,-15,-24,72,32,-22,-23,74,32
             ]
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
        ball_memory=ball_memory[5:45]
        if len(ball_memory)<40:
            continue
        # print(len(ball_memory))
        if len(ball_memory)==0:
            continue

        observations=ball_memory
        observations[:,3]=observations[:,3]-observations[0,3]

        arr_3d = observations.reshape(-1, 10,  observations.shape[1])  # -> shape (4, 5, 4)

        # 2. 对每个块的 5 行做列均值
        arr_means = arr_3d.mean(axis=1)
        observations=arr_means
        # print(observations)

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
    draw_multi_3D_trajetory_cls(raw_data=exp_results)

if __name__ == "__main__":
    main()
    # print(cal_position(14))