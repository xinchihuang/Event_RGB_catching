import plotly.graph_objects as go

import cv2
import numpy as np
from fitting import fit_3d_trajectory_with_distortion
# from scratch_ukf import observations
from utils import rotate_matrix_coordinate
import os
import time



def cal_position(point_index,y_bias=0,z_bias=0.8):
    row=point_index//10
    col=point_index%10
    y=(row)*0.1+y_bias+0.05
    z=(col)*0.1+z_bias+0.05
    return y,z
def compute_distance_stats(array1, array2):
    # 确保输入数组是二维的
    if array1.ndim == 1:
        array1 = array1.reshape(1, -1)
    if array2.ndim == 1:
        array2 = array2.reshape(1, -1)

    # 计算所有点对之间的欧式距离
    diff = array1[:, np.newaxis, :] - array2[np.newaxis, :, :]
    squared_diff = diff ** 2
    sum_squared_diff = np.sum(squared_diff, axis=2)
    distances = np.sqrt(sum_squared_diff)

    # 计算均值和方差
    mean = np.mean(distances)
    variance = np.var(distances)

    return mean, variance


def main():
    """ Main """
    plot_index=1
    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels_3D"
    num_file = len(
        [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
    # 创建图形对象
    fig = go.Figure()
    # camera_config = {
    #             'fx': 668.3,
    #             'fy': 668.3,
    #             'cx': 360.6,
    #             'cy': 640.6,
    #             'dist_'
    #             # 'oeffs': [-0.269418, 0.048704, -.000727, 0.002795, 0]  # k1, k2, p1, p2, k3
    #             'dist_coeffs':[ -2.9322907379719421e-01, 5.5767700541466243e-02,
    #             4.5890654823750650e-03, 5.7787331135171320e-05, 0.0 ]
    #         }
    camera_config = {
        'fx': 739.9,
        'fy': 739.9,
        'cx': 360,
        'cy': 720,
        'dist_'
        # 'oeffs': [-0.269418, 0.048704, -.000727, 0.002795, 0]  # k1, k2, p1, p2, k3
        'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]

    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])


    # 初始猜测参数（需要根据实际情况调整）
    init_guess = [3.5, 0, 1.5, -7, 0, 1]
    # label=[12,13,8,5,33,
    #        60,6,4,17,27,
    #        6,24,17,1,17,
    #        38,3,20,1,1,1]
    label = [-63]
    impact_point_real=[]
    impact_point_predict=[]
    for i in range(0,plot_index):
        # print(i)
        ball_memory=np.load(f"{save_dir}/{i}.npy")
        ball_memory=ball_memory[:]
        # print(i)
        if len(ball_memory)==0:
            # print(i)
            continue

        observations=ball_memory
        observations[:,3]=observations[:,3]-observations[0,3]
        # print(observations)
        result = fit_3d_trajectory_with_distortion(
            observations=observations,
            camera_params=camera_config,
            rotation_matrix=rotation_matrix_back,
        )
        x0=result['params']['x0']
        y0=result['params']['y0']
        z0=result['params']['z0']
        vx=result['params']['vx']
        vy=result['params']['vy']
        vz=result['params']['vz']
        print(x0,y0,z0,vx,vy,vz)
        impact_t=abs(x0/vx)
        impact_y=y0+vy*impact_t
        impact_z=z0+vz*impact_t-0.5*9.81*impact_t**2
        impact_point_predict.append([impact_y,impact_z])
        # print(impact_t,impact_y,impact_z,)
        t = np.linspace(0, 0.3, 100)
        # x=x0+vx*t
        # y=y0+vy*t
        # z=z0+vz*t-0.5*9.81*t**2
        ball=result['points']
        x=ball[:,0]
        y=ball[:,1]
        z=ball[:,2]
        # print("params",x0,y0,z0,vx,vy,vz)
        label_y,label_z=cal_position(label[i])
        impact_point_real.append([label_y,label_z])

        # fig.add_trace(go.Scatter3d(
        #     x=[0],  # X坐标固定为0
        #     y=[impact_y],
        #     z=[impact_z],
        #     mode='markers',
        #     name=f'predicted {i}',
        #     marker=dict(
        #         size=8,
        #         color='red',  # 预测点颜色
        #         symbol='circle'  # 圆形标记
        #     ),
        #     showlegend=True
        # ))
        # fig.add_trace(go.Scatter3d(
        #     x=[0],  # X坐标固定为0
        #     y=[label_y],
        #     z=[label_z],
        #     mode='markers',
        #     name=f'real {i}',
        #     marker=dict(
        #         size=8,
        #         color='green',  # 预测点颜色
        #         symbol='circle'  # 圆形标记
        #     ),
        #     showlegend=True
        # ))

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'observation {i}',
            line=dict(width=2)
        ))
        #
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(title='X', range=[0,4],dtick=1),
        #         yaxis=dict(title='Y', range=[-1,1],dtick=1),
        #         zaxis=dict(title='Z', range=[0,3],dtick=1),
        #         camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))  # 设定视角
        #
        #     ),
        #     margin=dict(l=0, r=0, b=0, t=30)
        # )
        # 显示图表
    fig.add_trace(go.Scatter3d(
        x=[0],  # X坐标固定为0
        y=[0],
        z=[0],
        mode='markers',
        name=f'camera',
        marker=dict(
            size=20,
            color='yellow',  # 预测点颜色
            symbol='circle'  # 圆形标记
        ),
        showlegend=True
    ))
    fig.show()
    print(compute_distance_stats(np.array(impact_point_real), np.array(impact_point_predict)))
if __name__ == "__main__":
    main()