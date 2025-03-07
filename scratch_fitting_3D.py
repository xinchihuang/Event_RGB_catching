import plotly.graph_objects as go

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig, draw_tracking_results
from metavision_sdk_core import BaseFrameGenerationAlgorithm, RollingEventBufferConfig, RollingEventCDBuffer
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from sklearn.cluster import DBSCAN
from fitting import fit_3d_trajectory_with_distortion
# from scratch_ukf import observations
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



def rotate_matrix_coordinate(r_x, r_y, r_z):
    """
    Right hand coordinate rotation matrix for coordinate rotation
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


def parse_args(i):
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generic Tracking sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
        '-i', '--input-event-file', dest='event_file_path',
        default=f"/home/xinchi/Event_RGB_catching_data/saved_data/clipped_event/{i}.raw",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')

    parser.add_argument('--update-frequency', dest='update_frequency', type=float,
                        default=10000., help="Tracker's update frequency, in Hz.")
    parser.add_argument('-a', '--acc-time', dest='accumulation_time_us', type=int, default=1000,
                        help='Duration of the time slice to store in the rolling event buffer at each tracking step')

    # Min/Max size options
    minmax_size_options = parser.add_argument_group('Min/Max size options')
    minmax_size_options.add_argument('--min-size', dest='min_size', type=int,
                                     default=40, help='Minimal size of an object to track (in pixels).')
    minmax_size_options.add_argument('--max-size', dest='max_size', type=int,
                                     default=50, help='Maximal size of an object to track (in pixels).')

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument(
        '--activity-time-ths', dest='activity_time_ths', type=int, default=10000,
        help='Length of the time window for activity filtering (in us, disabled if equal to 0).')
    filter_options.add_argument('--activity-trail-ths', dest='activity_trail_ths', type=int, default=10000,
                                help='Length of the time window for trail filtering (in us, disabled if equal to 0).')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video. A frame is generated every time the tracking callback is called.")

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
        '-f', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    # print(parser)
    args = parser.parse_args()

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)
    # print(args)
    return args


def get_i_ll_biases():
    """获取ILL偏置值"""
    try:
        import metavision_hal
        from metavision_hal import DeviceDiscovery, I_LL_Biases

        # 发现并打开设备
        device = DeviceDiscovery.open('')
        if device is None:
            print("未找到设备")
            return None

        # 获取ILL偏置接口
        i_ll_biases = device.get_facility(I_LL_Biases)
        if i_ll_biases is None:
            print("设备不支持ILL偏置")
            return None

        # 直接获取所有偏置值
        return i_ll_biases.get_all_biases()

    except Exception as e:
        print(f"获取ILL偏置时出错: {str(e)}")
        return None


def set_i_ll_bias(bias_name, bias_value):
    """设置指定的ILL偏置值"""
    try:
        import metavision_hal
        from metavision_hal import DeviceDiscovery, I_LL_Biases

        device = DeviceDiscovery.open('')
        if device is None:
            print("未找到设备")
            return False

        i_ll_biases = device.get_facility(I_LL_Biases)
        if i_ll_biases is None:
            print("设备不支持ILL偏置")
            return False

        # 设置偏置值
        success = i_ll_biases.set(bias_name, bias_value)
        if success:
            # 获取并显示新的值
            new_value = i_ll_biases.get(bias_name)
            print(f"成功设置 {bias_name} = {new_value}")
        return success

    except Exception as e:
        print(f"设置ILL偏置时出错: {str(e)}")
        return False


def main():
    """ Main """
    # biases = get_i_ll_biases()
    # if biases:
    #     print("当前ILL偏置值:")
    #     for name, value in biases.items():
    #         print(f"{name}: {value}")
    #         # 可选：获取每个偏置的详细信息
    #         # bias_info = i_ll_biases.get_bias_info(name)
    plot_index=1
    #
    #
    # for i in range(0,plot_index):
    #     from metavision_core.event_io import EventsIterator
    #
    #     event_iterator = iter(EventsIterator(input_path=f"/home/xinchi/Event_RGB_catching_data/saved_data/event/{i}.raw", mode="mixed",
    #                        delta_t=10000, n_events=10000))
    #
    #     # Camera Related
    #     # 定义输入参数
    #     camera_config = {
    #         'fx': 668.3,
    #         'fy': 668.3,
    #         'cx': 310.6,
    #         'cy': 630.6,
    #         'dist_'
    #         # 'oeffs': [-0.269418, 0.048704, -.000727, 0.002795, 0]  # k1, k2, p1, p2, k3
    #         'coeffs':[ -2.9322907379719421e-01, 5.5767700541466243e-02,
    #         4.5890654823750650e-03, 5.7787331135171320e-05, 0.0 ]
    #     }
    #
    #     # 单位矩阵表示世界坐标系与相机坐标系对齐
    #     camera_rpy_to = [0, 45, -90]
    #     camera_rpy_back = [45, 0, 90]
    #     rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    #     rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])
    #     ball_memory=np.zeros((10000,4))
    #     count=0
    #     while True:
    #
    #         evs=next(event_iterator)
    #         if len(evs)==0:
    #             break
    #         evs['x'], evs['y'] = evs['y'].copy(), -evs['x'].copy()
    #         evs['y'] += 1279
    #         event_data = evs
    #         event_width, event_height = 720, 1280
    #         event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_width, sensor_height=event_height,
    #                                                            fps=100,
    #                                                            palette=ColorPalette.Dark)
    #         event_frame = np.zeros((event_height, event_width, 3), dtype=np.uint8)
    #         event_frame = np.ascontiguousarray(event_frame)
    #         event_frame_gen.generate_frame(event_data, event_frame)
    #         image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
    #         _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #         points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
    #         if points.shape[0] == 0:
    #             continue
    #         dbscan = DBSCAN(eps=5, min_samples=30).fit(points)
    #         labels = dbscan.labels_
    #         color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #         for cluster_id in set(labels):
    #             if cluster_id == -1:  # -1 代表噪声点，不处理
    #                 continue
    #             cluster_points = points[labels == cluster_id]
    #             x, y, w, h = cv2.boundingRect(cluster_points)
    #             cv2.rectangle(color_image, (y, x), (y + h, x + w), (0, 255, 0), 2)
    #             # print(h)
    #
    #
    #             ball_memory[count][0]=x
    #             ball_memory[count][1]=y
    #             ball_memory[count][2]=h
    #             ball_memory[count][3]=time.time()
    #             # ball_memory.append([x,y,h,time.time()])
    #
    #             break
    #         cv2.imshow("image", color_image)
    #         count += 1
    #         print(count)
    #         cv2.waitKey(1)
    #
    #     ball_memory_array=ball_memory[~np.all(ball_memory == 0, axis=1)]
    #     # print(ball_memory)
    #     save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels_3D"
    #     num_file = len(
    #         [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
    #     save_path = f"{save_dir}/{i}.npy"
    #     # print(ball_memory)
    #     print("save to:", save_path)
    #     np.save(save_path, ball_memory)





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