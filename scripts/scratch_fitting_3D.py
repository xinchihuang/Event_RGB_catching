import numpy as np
from fitting import fit_3d_trajectory_with_distortion
from utils import rotate_matrix_coordinate
import os
from plot.plot_3D import draw_3D_trajetory,draw_multi_3D_trajetory,draw_multi_3D_trajetory_cls
from parabola_fitter import MonoPixelFitter, MonoFitter, MonoFitter3D,MonoPixelFitter3D,MonoRansac


def index_to_position(point_index,y_bias=0,z_bias=0.2):
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

def position_to_index(position, y_bias=0, z_bias=0.2):
    """Convert a 3D position to a grid index."""
    y = position[1] - y_bias - 0.05
    z = position[2] - z_bias - 0.05
    row = int(z // 0.1)
    col = int(y // 0.1)
    print("row:", row, "col:", col, "y:", y, "z:", z)
    if y>=0:
        return -row * 10 + col
    else:
        return row * 10 - col


def main():
    """ Main """

    plot_index=29
    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixel_3D_records_model_c"
    # camera_config = {
    #     'fx': 739.9,
    #     'fy': 739.9,
    #     'cx': 310,
    #     'cy': 710,
    #     'dist_'
    #     'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]
    # }


    camera_config = {
        "fx": 987.1678420424932,
        "fy": 986.0683237433913,
        "cy": 739.7418995490075,
        "cx": 513.7733221750243,
        "dist_coeffs": [
            -0.11123899870851763,
            -0.020663002330730863,
            -0.0002997445722290338,
            0.0007964973013426446,
            0.014818755754515196
        ]
    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])

    label_file= "/home/xinchi/Event_RGB_catching/fitting/label_c.csv"
    labels=np.loadtxt(label_file, delimiter=",")
    print(labels)
    # fitter = MonoFitter3D(camera_params=camera_config,rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter(camera_params=camera_config, rotation_matrix_to=rotation_matrix_to,
    #                          rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter3DRansac(camera_params=camera_config, rotation_matrix_back=rotation_matrix_back)
    # fitter = MonoPixelFitter(camera_params=camera_config, rotation_matrix_back=rotation_matrix_back)
    # fitter = PixelSizeFitter(camera_params=camera_config,rotation_matrix_back=rotation_matrix_back)
    fitter = MonoRansac(camera_params=camera_config,rotation_matrix_to=rotation_matrix_to,rotation_matrix_back=rotation_matrix_back)
    exp_results = []

    for i in range(0,100):
        # Skip invalid data where label is 0
        if labels[i] == 0:
            print(f"Skipping trajectory {i}: invalid label (0)")
            continue
            
        ball_memory=np.loadtxt(f"{save_dir}/{i}.csv", delimiter=",")
        print(ball_memory)
        print(i)
        observations=ball_memory
        observations[:,3]=observations[:,3]-observations[0,3]
        observations= observations[:20, :]
        fitter.set_data(observations)
        result=fitter.get_params()

        impact_point_real=index_to_position(labels[i])
        ball = result['points']

        impact_point=fitter.predict()

        dict_to_save={'points' : ball,
             'impact':impact_point,
             'label':impact_point_real,
             'fit_params': result.get('params', None),
             'inliers': result.get('inliers', None),
             'trajectory_id': i,  # Add original trajectory ID
             }
        exp_results.append(dict_to_save)
    draw_multi_3D_trajetory_cls(raw_data=exp_results)

if __name__ == "__main__":
    main()
    # print(cal_position(14))