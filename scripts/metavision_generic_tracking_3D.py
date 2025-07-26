# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track general objects.
You can use it, for example, with the reference file traffic_monitoring.raw.
"""

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig, draw_tracking_results
from metavision_sdk_core import BaseFrameGenerationAlgorithm, RollingEventBufferConfig, RollingEventCDBuffer
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
#!/usr/bin/env python3
import math
import numpy as np
import cv2
import time
from parabola_fitter import *
import can
from pcan_cybergear import CANMotorController
def update_states(robot_pose,ball_memory,move_msg,bbox):
    cx=619.674
    cy=333.682
    fx=673.797
    fy=673.797
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    distort_matrix = np.array([-0.269418, 0.048704, -0.000727, 0.002795, 0])
    present_time = time.time()
    time_step = present_time - robot_pose[3]
    robot_pose = [robot_pose[0] + move_msg[0] * time_step,
                       robot_pose[1] + move_msg[1] * time_step,
                       robot_pose[2] + move_msg[2] * time_step, robot_pose[3] + time_step]


    distorted_pixel = np.array([bbox[0], bbox[1]], dtype=np.float32).reshape(1, 1, 2)

    # Undistort the pixel
    undistorted_pixel = cv2.undistortPoints(distorted_pixel, intrinsic_matrix, distort_matrix, P=intrinsic_matrix)
    x_cp = undistorted_pixel[0][0][0]
    y_cp = undistorted_pixel[0][0][1]
    x_c = (x_cp - cx) / fx
    y_c = (y_cp - cy) / fy
    #process event_camera rotation
    x_c,y_c = y_c,-x_c
    # print(x_c,y_c)
    if len(ball_memory) > 100:
        ball_memory.pop(0)

    if len(ball_memory)==0 or not (x_c==ball_memory[-1][0] and y_c==ball_memory[-1][1]):
        ball_memory.append([x_c, y_c, present_time])

    return ball_memory,robot_pose


def ball_tracking_simple(bbox):
    adjust_d=-150
    cx = 619.674
    cy = 333.682
    fx = 673.797
    fy = 673.797
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    distort_matrix = np.array([-0.269418, 0.048704, -0.000727, 0.002795, 0])
    present_time = time.time()
    distorted_pixel = np.array([bbox[0][0], bbox[0][1]], dtype=np.float32).reshape(1, 1, 2)

    # Undistort the pixel
    undistorted_pixel = cv2.undistortPoints(distorted_pixel, intrinsic_matrix, distort_matrix, P=intrinsic_matrix)
    x_cp = undistorted_pixel[0][0][0]
    y_cp = undistorted_pixel[0][0][1]
    # print(x_cp,y_cp,x_cp-cx,y_cp-cy)
    target_theta=math.atan2((x_cp-cx-adjust_d),(y_cp-cy))

    if (x_cp-cx-adjust_d)<0:
        target_theta=0
    return (90-math.degrees(target_theta))/abs(90-math.degrees(target_theta))*min(45,abs(90-math.degrees(target_theta)))



def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generic Tracking sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    #/home/yifan/Documents/metavision/recordings/recording_2024-12-09_15-44-35.raw"
    base_options.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="/home/xinchi/programs/event camera recordings/recording_2024-12-09_15-44-35.raw",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')

    parser.add_argument('--update-frequency', dest='update_frequency', type=float,
                        default=100., help="Tracker's update frequency, in Hz.")
    parser.add_argument('-a', '--acc-time', dest='accumulation_time_us', type=int, default=1000,
                        help='Duration of the time slice to store in the rolling event buffer at each tracking step')

    # Min/Max size options
    minmax_size_options = parser.add_argument_group('Min/Max size options')
    minmax_size_options.add_argument('--min-size', dest='min_size', type=int,
                                     default=30, help='Minimal size of an object to track (in pixels).')
    minmax_size_options.add_argument('--max-size', dest='max_size', type=int,
                                     default=100, help='Maximal size of an object to track (in pixels).')

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument(
        '--activity-time-ths', dest='activity_time_ths', type=int, default=100,
        help='Length of the time window for activity filtering (in us, disabled if equal to 0).')
    filter_options.add_argument('--activity-trail-ths', dest='activity_trail_ths', type=int, default=1000,
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


    # bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
    # # bus.shutdown()
    # motor1 = CANMotorController(bus, motor_id=101, main_can_id=254)
    #
    # motor1.set_0_pos()
    #
    # motor1.enable()
    #
    # motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0, Kp=1, Kd=1)
    # time.sleep(2.5)
    # motor1.disable()
    #
    # bus.shutdown()

    """ Main """
    biases = get_i_ll_biases()
    if biases:
        print("当前ILL偏置值:")
        for name, value in biases.items():
            print(f"{name}: {value}")
            # 可选：获取每个偏置的详细信息
            # bias_info = i_ll_biases.get_bias_info(name)
    
    args = parse_args()
    # print(args)
    # print(type(args))

    # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_BEGIN]
    # Rolling event buffer
    buffer_config = RollingEventBufferConfig.make_n_us(args.accumulation_time_us)
    rolling_buffer = RollingEventCDBuffer(buffer_config)
    # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_END]

    # [GENERIC_TRACKING_CREATE_ITERATOR_BEGIN]
    # Events iterator on Camera or event file
    delta_t = int(1000000 / args.update_frequency)
    mv_iterator = EventsIterator(input_path=args.event_file_path, start_ts=args.process_from,
                                 max_duration=args.process_to - args.process_from if args.process_to else None,
                                 delta_t=delta_t, mode="delta_t")
    # [GENERIC_TRACKING_CREATE_ITERATOR_END]

    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

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
    

    # [GENERIC_TRACKING_MAIN_PROCESSING_BEGIN]
    def process_tracking(evs):
        # print(evs)
        if len(evs) != 0:
            rolling_buffer.insert_events(evs)
            tracking_algo.process_events(rolling_buffer, tracking_results)
            BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
            # if len(tracking_results.numpy()) > 0:
            #     print(tracking_results.numpy())
            draw_tracking_results(evs['t'][-1], tracking_results, output_img)
            
        window.show_async(output_img)
        if args.out_video:
            video_writer.write(output_img)

    def process_tracking_3D_simple(evs,ball_memory,robot_pose,move_msg):
        # print(evs)
        impact_point=[0,0,0]
        target_theta=-8080
        if len(evs) != 0:
            rolling_buffer.insert_events(evs)
            tracking_algo.process_events(rolling_buffer, tracking_results)
            if len(tracking_results.numpy()) == 1:
                # print(tracking_results.numpy())
                bbox=tracking_results.numpy()
                target_theta=ball_tracking_simple(bbox)
                # bbox=[50,50]
                # robot_pose, ball_memory = update_states(robot_pose, ball_memory, move_msg, bbox)
                # move_msg,impact_point=ball_tracking(ball_memory,robot_pose)
                # print(impact_point)
            BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
            draw_tracking_results(evs['t'][-1], tracking_results, output_img)
        window.show_async(output_img)
        # return robot_pose,ball_memory,move_msg,impact_point
        return -target_theta
    def process_tracking_3D(evs,ball_memory,robot_pose,move_msg,fitter=None):
        # print(evs)
        impact_point=[0,0,0]
        target_theta=-8080
        if len(evs) != 0:
            rolling_buffer.insert_events(evs)
            tracking_algo.process_events(rolling_buffer, tracking_results)
            if len(tracking_results.numpy()) == 1:
                # print(tracking_results.numpy())
                bbox=tracking_results.numpy()
                # print(bbox)
                # target_theta=ball_tracking_simple(bbox)

                bbox=[tracking_results.numpy()[0][0],tracking_results.numpy()[0][1]]
                # print(bbox)
                ball_memory,robot_pose, = update_states(robot_pose, ball_memory, move_msg, bbox)
                # print(ball_memory[-1],robot_pose)
                params=fitter.mono_fitter(ball_memory)
                # print(params)
                if not params == None:
                    x_0, y_0, z_0, vx, vy, vz=params
                    impact_t=x_0/vx
                    print(x_0,vx,impact_t)
              # move_msg,impact_point=ball_tracking(ball_memory,robot_pose)
                # print(impact_point)
            BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
            draw_tracking_results(evs['t'][-1], tracking_results, output_img)
        window.show_async(output_img)
        # return robot_pose,ball_memory,move_msg,impact_point
        return -target_theta
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
        ball_memory=[]
        robot_pose=[0,0,0,0]
        move_msg=[0,0,0]
        buffer_size=2000
        target_theta_buffer=[]
        parabola_fitter=ParabolaFitter(camera_rpy_to=[0,45,-90],camera_rpy_back=[-45,0,90])
        for evs in mv_iterator:

            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            if args.activity_time_ths > 0:
                activity_noise_filter.process_events(evs, events_buf)
                if args.activity_trail_ths > 0:
                    trail_filter.process_events_(events_buf)

                # process_tracking(events_buf.numpy())
                # robot_pose,ball_memory,move_msg,impact_point=process_tracking_3D(events_buf.numpy(),ball_memory,robot_pose,move_msg)
                target_theta=process_tracking_3D(events_buf.numpy(),ball_memory,robot_pose,move_msg,parabola_fitter)
            elif args.activity_trail_ths > 0:
                trail_filter.process_events(evs, events_buf)

                # process_tracking(events_buf.numpy())
                # robot_pose,ball_memory,move_msg,impact_point=process_tracking_3D(events_buf.numpy(),ball_memory,robot_pose,move_msg)
                target_theta=process_tracking_3D(events_buf.numpy(),ball_memory,robot_pose,move_msg,parabola_fitter)
            else:
                # process_tracking(evs)
                # robot_pose,ball_memory,move_msg,impact_point=process_tracking_3D(evs,ball_memory,robot_pose,move_msg)
                target_theta=process_tracking_3D(evs,ball_memory,robot_pose,move_msg,parabola_fitter)

            # if len(target_theta_buffer)>30:
            #     target_theta_buffer.pop(0)
            if not target_theta==8080 and len(target_theta_buffer)<buffer_size:
                target_theta_buffer.append(target_theta)
            if len(target_theta_buffer)==buffer_size:
                target_theta_mean=sum(target_theta_buffer)/len(target_theta_buffer)
                print(target_theta_mean)
                # motor1.send_motor_control_command(torque=0, target_angle=math.radians(target_theta_mean), target_velocity=0,
                #                                   Kp=100, Kd=1)
                break

            if window.should_close():
                break
    # time.sleep(2)
    # motor1.send_motor_control_command(torque=0, target_angle=0,
    #                                   target_velocity=0,
    #                                   Kp=10, Kd=1)
    # time.sleep(2)
    # motor1.disable()
    #
    # bus.shutdown()

if __name__ == "__main__":
    main()
