import plotly.graph_objects as go

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig, draw_tracking_results
from metavision_sdk_core import BaseFrameGenerationAlgorithm, RollingEventBufferConfig, RollingEventCDBuffer
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
# from ekf_predict import ekf_predict
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

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track general objects.
You can use it, for example, with the reference file traffic_monitoring.raw.
"""




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


    for i in range(0,plot_index):
        args = parse_args(i)
        # print(args)
        # print(type(args))

        # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_BEGIN]
        # Rolling event buffer
        buffer_config = RollingEventBufferConfig.make_n_us(args.accumulation_time_us)
        rolling_buffer = RollingEventCDBuffer(buffer_config)
        # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_END]

        # [GENERIC_TRACKING_CREATE_ITERATOR_BEGIN]
        # Events iterator on Camera or event file
        delta_t = int(100000 / args.update_frequency)
        # print(delta_t)
        mv_iterator = EventsIterator(input_path=args.event_file_path, start_ts=args.process_from,
                                     max_duration=args.process_to - args.process_from if args.process_to else None,
                                     delta_t=delta_t, mode="delta_t")
        # [GENERIC_TRACKING_CREATE_ITERATOR_END]

        if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
            mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
        # height, width = mv_iterator.get_size()  # Camera Geometry
        width, height= mv_iterator.get_size()  # Camera Geometry
        # Noise + Trail filter that will be applied to events
        activity_noise_filter = ActivityNoiseFilterAlgorithm(width, height, args.activity_time_ths)
        trail_filter = TrailFilterAlgorithm(width, height, args.activity_trail_ths)

        # Tracking Algorithm
        tracking_config = TrackingConfig()  # Default configuration
        tracking_algo = TrackingAlgorithm(sensor_width=width, sensor_height=height, tracking_config=tracking_config)
        tracking_algo.min_size = args.min_size
        tracking_algo.max_size = args.max_size

        output_img = np.zeros((height, width, 3), np.uint8)

        events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
        tracking_results = tracking_algo.get_empty_output_buffer()

        # Camera Related
        # 定义输入参数
        camera_config = {
            'fx': 668.3,
            'fy': 668.3,
            'cx': 310.6,
            'cy': 630.6,
            'dist_'
            # 'oeffs': [-0.269418, 0.048704, -.000727, 0.002795, 0]  # k1, k2, p1, p2, k3
            'coeffs':[ -2.9322907379719421e-01, 5.5767700541466243e-02,
            4.5890654823750650e-03, 5.7787331135171320e-05, 0.0 ]
        }

        # 单位矩阵表示世界坐标系与相机坐标系对齐
        camera_rpy_to = [0, 45, -90]
        camera_rpy_back = [45, 0, 90]
        rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
        rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])


        # [GENERIC_TRACKING_MAIN_PROCESSING_BEGIN]

        def process_tracking(evs,ball_memory):
            if len(evs) != 0:
                rolling_buffer.insert_events(evs)
                tracking_algo.process_events(rolling_buffer, tracking_results)
                BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
                if len(tracking_results.numpy()) == 1:
                   # print(tracking_results.numpy()[0][2]/1000000)
                   ball_memory.append([tracking_results.numpy()[0][0],tracking_results.numpy()[0][1],tracking_results.numpy()[0][2]/1000000])

                draw_tracking_results(evs['t'][-1], tracking_results, output_img)
            window.show_async(output_img)
            if args.out_video:
                video_writer.write(output_img)
            return ball_memory

        # [GENERIC_TRACKING_MAIN_PROCESSING_END]

        # Window - Graphical User Interface (Display tracking results and process keyboard events)
        with MTWindow(title="Generic Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
            window.show_async(output_img)

            if args.out_video:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video_name = args.out_video + ".avi"
                video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

            def keyboard_cb(key, scancode, action, mods):
                SIZE_STEP = 10

                if action != UIAction.RELEASE:
                    return
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    window.set_close_flag()
                # A: Increase minimum size of the object to track
                elif key == UIKeyEvent.KEY_A:
                    if args.min_size + SIZE_STEP <= args.max_size:
                        args.min_size += SIZE_STEP
                        print("Increase min size to {}".format(args.min_size))
                        tracking_algo.min_size = args.min_size
                # B: Decrease minimum size of the object to track
                elif key == UIKeyEvent.KEY_B:
                    if args.min_size - SIZE_STEP >= 0:
                        args.min_size -= SIZE_STEP
                        print("Decrease min size to {}".format(args.min_size))
                        tracking_algo.min_size = args.min_size
                # C: Increase maximum size of the object to track
                elif key == UIKeyEvent.KEY_C:
                    args.max_size += SIZE_STEP
                    print("Increase max size to {}".format(args.max_size))
                    tracking_algo.max_size = args.max_size
                # D: Decrease maximum size of the object to track
                elif key == UIKeyEvent.KEY_D:
                    if args.max_size - SIZE_STEP >= args.min_size:
                        args.max_size -= SIZE_STEP
                        print("Decrease max size to {}".format(args.max_size))
                        tracking_algo.max_size = args.max_size

            window.set_keyboard_callback(keyboard_cb)
            print("Press 'q' to leave the program.\n"
                  "Press 'a' to increase the minimum size of the object to track.\n"
                  "Press 'b' to decrease the minimum size of the object to track.\n"
                  "Press 'c' to increase the maximum size of the object to track.\n"
                  "Press 'd' to decrease the maximum size of the object to track.")

            # [GENERIC_TRACKING_MAIN_LOOP_BEGIN]
            # Process events
            ball_memory = []
            for evs in mv_iterator:

                # print(count)
                # count += 1
                # if count>500:
                #     break

                evs['x'], evs['y'] = evs['y'].copy(), -evs['x'].copy()
                evs['y'] += 1279
                # print(time.time())
                # Dispatch system events to the window
                EventLoop.poll_and_dispatch()

                # Process events
                if args.activity_time_ths > 0:
                    activity_noise_filter.process_events(evs, events_buf)
                    if args.activity_trail_ths > 0:
                        trail_filter.process_events_(events_buf)

                    process_tracking(events_buf.numpy(), ball_memory)

                elif args.activity_trail_ths > 0:
                    trail_filter.process_events(evs, events_buf)

                    process_tracking(events_buf.numpy(), ball_memory)
                else:
                    process_tracking(evs,ball_memory)

                if window.should_close():
                    break
            save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels"
            num_file = len(
                [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
            save_path = f"{save_dir}/{i}.npy"
            print(ball_memory)
            print("save to:", save_path)
            np.save(save_path, ball_memory)

            # [GENERIC_TRACKING_MAIN_LOOP_END]

            if args.out_video:
                video_writer.release()
                print("Video has been saved in " + video_name)




    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels"
    num_file = len(
        [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
    # 创建图形对象
    fig = go.Figure()
    camera_config = {
                'fx': 668.3,
                'fy': 668.3,
                'cx': 310.6,
                'cy': 630.6,
                'dist_'
                # 'oeffs': [-0.269418, 0.048704, -.000727, 0.002795, 0]  # k1, k2, p1, p2, k3
                'dist_coeffs':[ -2.9322907379719421e-01, 5.5767700541466243e-02,
                4.5890654823750650e-03, 5.7787331135171320e-05, 0.0 ]
            }
    # 单位矩阵表示世界坐标系与相机坐标系对齐
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])


    # 初始猜测参数（需要根据实际情况调整）
    init_guess = [3.5, 0, 1.5, -7, 0, 1]
    label=[12,13,8,5,33,
           60,6,4,17,27,
           6,24,17,1,17,
           38,3,20,1,1,1]
    label = [-42, 21, -31, 64, 61,
             32, 44, 65, 14, 15,
             34, -55, 62, 33, -24,
             -13, -65, -62, -53, -32]
    impact_point_real=[]
    impact_point_predict=[]
    for i in range(0,plot_index):
        print(i)
        ball_memory=np.load(f"{save_dir}/{i}.npy")
        # print(ball_memory)
        ball_memory=ball_memory[5:-5]
        # print(i)
        if len(ball_memory)==0:
            # print(i)
            continue
        u=ball_memory[:,0]
        v=ball_memory[:,1]
        t=ball_memory[:,2]

        observations=ball_memory
        observations[:,2]=observations[:,2]-observations[0,2]
        # print(observations)
        result = fit_3d_trajectory_with_distortion(
            observations=observations,
            camera_params=camera_config,
            rotation_matrix=rotation_matrix_to,
            initial_guess=init_guess
        )
        x0=result['params']['x0']
        y0=result['params']['y0']
        z0=result['params']['z0']
        vx=result['params']['vx']
        vy=result['params']['vy']
        vz=result['params']['vz']
        impact_t=abs(x0/vx)
        impact_y=y0+vy*impact_t
        impact_z=z0+vz*impact_t-0.5*9.81*impact_t**2
        impact_point_predict.append([impact_y,impact_z])
        # print(impact_t,impact_y,impact_z,)
        t = np.linspace(0, 0.3, 100)
        x=x0+vx*t
        y=y0+vy*t
        z=z0+vz*t-0.5*9.81*t**2
        # print("params",x0,y0,z0,vx,vy,vz)
        label_y,label_z=cal_position(label[i])
        impact_point_real.append([label_y,label_z])

        fig.add_trace(go.Scatter3d(
            x=[0],  # X坐标固定为0
            y=[impact_y],
            z=[impact_z],
            mode='markers',
            name=f'predicted {i}',
            marker=dict(
                size=8,
                color='red',  # 预测点颜色
                symbol='circle'  # 圆形标记
            ),
            showlegend=True
        ))
        fig.add_trace(go.Scatter3d(
            x=[0],  # X坐标固定为0
            y=[label_y],
            z=[label_z],
            mode='markers',
            name=f'real {i}',
            marker=dict(
                size=8,
                color='green',  # 预测点颜色
                symbol='circle'  # 圆形标记
            ),
            showlegend=True
        ))

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'observation {i}',
            line=dict(width=2)
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', range=[0,4],dtick=1),
                yaxis=dict(title='Y', range=[-1,1],dtick=1),
                zaxis=dict(title='Z', range=[0,3],dtick=1),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))  # 设定视角

            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
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