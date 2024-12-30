import cv2

import argparse
import os

import matplotlib.pyplot as plt
import sys

import torch
from multiprocessing import Process,Array,Queue
import multiprocessing
import signal
import numpy as np
import time
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)




def transform_points_to_voxels(event_data, voxel_generator=None, device=torch.device("cuda")):
    """
    将点云转换为voxel,调用spconv的VoxelGeneratorV2
    """
    save_voxel = 5000
    x =event_data['x']  # Shape: (N,)
    # print(x)
    y =event_data['y']  # Shape: (N,)
    p =event_data['p']  # Shape: (N,)
    t =event_data['t']  # Shape: (N,)
    # print(t)

    time_length = t[-1] - t[0]
    # print(time_length)
    t = ((t - t[0]) / time_length) * 1000

    current_events = np.stack((t, x, y, p), axis=-1)
    current_events = torch.tensor(current_events,device=device).float()
    # print(current_events)
    voxel_output = voxel_generator(current_events)

    # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
    # voxels代表了每个生成的voxel数据，维度是[M, 5, 4]
    # coordinates代表了每个生成的voxel所在的zyx轴坐标，维度是[M,3]
    # num_points代表了每个生成的voxel中有多少个有效的点维度是[m,]，因为不满5会被0填充
    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel  8000 from(4k+,6k+)
    # print(torch.where(num_points>=16)[0].shape)
    if num_points.shape[0] < save_voxel:
        features = voxels[:, :, 3]
        coor = coordinates[:, :]
    else:
        _, voxels_idx = torch.topk(num_points, save_voxel)
        # 将每个voxel的1024个p拼接作为voxel初始特征   16
        features = voxels[voxels_idx][:, :, 3]
        # 前5000个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features

def read_from_event(event_queue, stop_event):
    from metavision_core.event_io import EventsIterator

    event_iterator = iter(EventsIterator(input_path="", delta_t=1000, n_events=100))
    while not stop_event.is_set():
        try:
            if event_queue.full():
                event_queue.get()  # Remove the oldest item
            event_data = next(event_iterator)
            event_queue.put(event_data, timeout=0.01)
        except Exception as e:
            print(f"Error in read_from_event: {e}")
            continue
def read_from_flir(flir_queue, stop_event):
    import PySpin
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()
    cam.BeginAcquisition()
    while not stop_event.is_set():
        try:
            # print('flir')
            image_result = cam.GetNextImage(10000)
            if image_result.IsIncomplete():
                continue
            image_data = image_result.GetNDArray()
            cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            if flir_queue.full():
                flir_queue.get()
            flir_queue.put(cv_image, timeout=0.1)
            image_result.Release()
        except Exception as e:
            print(f"Error in read_from_flir: {e}")
            continue


def model(event_queue, flir_queue, stop_event):
    from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
    from lib.test.evaluation.tracker import Tracker
    from spconv.pytorch.utils import PointToVoxel
    def draw_initial_bbox(image):
        # Convert from BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV
        lower_green = np.array([30, 40, 50])
        upper_green = np.array([60, 150, 255])

        # Create a mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Optional: Clean up the mask by removing noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Proceed only if at least one contour was found
        if len(contours) > 0:
            # Find the largest contour by area
            c = max(contours, key=cv2.contourArea)
            # Get bounding box coordinates from the largest contour
            x, y, w, h = cv2.boundingRect(c)
            if w < 100 or h < 100:
                return None
            return x, y, w, h
        else:
            return None
    device = torch.device("cuda")
    voxel_generator = PointToVoxel(
        # 给定每个voxel的长宽高  [0.05, 0.05, 0.1]
        vsize_xyz=[1000, 10, 10],  # [0.2, 0.25, 0.16]  # [50, 10, 10]  [50, 35, 26] 因此坐标范围（20,20,20）  (20, 34/35, 26)
        # 给定点云的范围 [  0.  -40.   -3.   70.4  40.    1. ]
        coors_range_xyz=[  0.,  0,   0,   100000,  1280 ,   720 ],
        # 给定每个点云的特征维度，这里是x，y，z，r 其中r是激光雷达反射强度       # 346x260  t,x,y
        num_point_features=4,
        # 最多选取多少个voxel，训练16000，推理40000
        max_num_voxels=16000,  # 16000
        # 给定每个pillar中有采样多少个点，不够则补0  因此我将neg voxel改为-1;
        max_num_points_per_voxel=16,  # 1024
        device=device
    )
    AMtracker = Tracker('amttrack', 'amttrack_felt', 'felt', None)

    params = AMtracker.get_parameters()
    # print(params)
    params.debug = None
    tracker = AMtracker.create_tracker(params)
    start_time = time.time()
    first_flir_image = flir_queue.get(timeout=0.1)
    event_template = torch.zeros([1024, 19])
    is_tracker_initialiezed=False
    initial_bbox = None

    with MTWindow(title="Metavision Events Viewer", width=1080, height=1440,
                  mode=BaseWindow.RenderMode.BGR) as window:

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=1280, sensor_height=720, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            cd_frame = cv2.rotate(cd_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)


        while not stop_event.is_set():
            # start_time = time.time()
            try:
                event_data = event_queue.get(timeout=0.1)
                # print(event_data.dtype)
                flir_data = flir_queue.get(timeout=0.1)
            except Exception as e:
                print(f"No Object")
                continue

            event_frame_gen.process_events(event_data)
            if not is_tracker_initialiezed:
                initial_bbox = draw_initial_bbox(flir_data)
                if initial_bbox is None:
                    pass
                else:
                    x, y, w, h = initial_bbox
                    bbox = [x,y,w,h]
                    cv2.rectangle(flir_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    init_info = {'init_bbox': bbox}
                    out = tracker.initialize(first_flir_image, event_template, init_info, idx=0)
                    is_tracker_initialiezed = True

            else:

                info = {'previous_output':{'target_bbox':bbox},'gt_bbox': bbox}
                # info['previous_output'] = None
                coor, features = transform_points_to_voxels(event_data=event_data, voxel_generator=voxel_generator,
                                                            device=device)
                # print(coor.shape, features.shape)
                event_search =  torch.cat((coor, features), dim=1)
                if event_search==None:
                    continue
                tracker.state=bbox
                out = tracker.track(flir_data, event_search, info)
                # print("OUT",out)
                bbox=out['target_bbox']

                state = [int(s) for s in bbox]
                ROI_queue.put(state,timeout=0.01)
                cv2.rectangle(flir_data, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),color=(0, 255, 0), thickness=2)

            # Display the image
            cv2.imshow('Image with Bounding Box', flir_data)
            cv2.waitKey(1)
            # end_t = time.time()
            # print("fps:",1/(end_t - start_time))

def pure_event_tracking(ROI_queue, event_queue):


    from metavision_core.event_io import EventsIterator
    from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
    from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig, draw_tracking_results
    from metavision_sdk_core import BaseFrameGenerationAlgorithm, RollingEventBufferConfig, RollingEventCDBuffer
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
    # Event tracker SDK args
    # This need to be carefully tuned to set good parameters
    # Some parameters that related to the events iterator should be modified in the read_from_event
    def parse_args():
        import argparse
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Generic Tracking sample.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Base options
        base_options = parser.add_argument_group('Base options')
        # /home/yifan/Documents/metavision/recordings/recording_2024-12-09_15-44-35.raw"
        base_options.add_argument(
            '-i', '--input-event-file', dest='event_file_path',
            default="/home/xinchi/programs/event camera recordings/recording_2024-12-09_15-44-35.raw",
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
    def ROI_filter(evs,ROI,outer_range=50):
        x,y,w,h = ROI
        evs_x,evs_y=evs
        if x-outer_range<evs_x < x+w+outer_range and y-outer_range<evs_y< y+h+outer_range:
            return True
        return

    args = parse_args()
    buffer_config = RollingEventBufferConfig.make_n_us(args.accumulation_time_us)
    rolling_buffer = RollingEventCDBuffer(buffer_config)
    # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_END]

    # [GENERIC_TRACKING_CREATE_ITERATOR_BEGIN]
    # Events iterator on Camera or event file
    # delta_t = int(1000000 / args.update_frequency)
    # mv_iterator = EventsIterator(input_path=args.event_file_path, start_ts=args.process_from,
    #                              max_duration=args.process_to - args.process_from if args.process_to else None,
    #                              delta_t=delta_t, mode="delta_t")
    # # [GENERIC_TRACKING_CREATE_ITERATOR_END]
    #
    # if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
    #     mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    # height, width = mv_iterator.get_size()  # Camera Geometry


    width = 1080
    height = 1440
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

    while not stop_event.is_set():
        # start_time = time.time()
        try:
            evs= event_queue.get(timeout=0.1)
            ROI= ROI_queue.get(timeout=0.1)
            if ROI_filter(evs,ROI)==False:
                continue
            EventLoop.poll_and_dispatch()
            # Process events
            if args.activity_time_ths > 0:
                activity_noise_filter.process_events(evs, events_buf)
                if args.activity_trail_ths > 0:
                    trail_filter.process_events_(events_buf)

                rolling_buffer.insert_events(evs)
                tracking_algo.process_events(rolling_buffer, tracking_results)
                BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
                if len(tracking_results.numpy()) > 0:
                    print(tracking_results.numpy())
            elif args.activity_trail_ths > 0:
                trail_filter.process_events(evs, events_buf)

                rolling_buffer.insert_events(evs)
                tracking_algo.process_events(rolling_buffer, tracking_results)
                BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
                if len(tracking_results.numpy()) > 0:
                    print(tracking_results.numpy())
            else:
                rolling_buffer.insert_events(evs)
                tracking_algo.process_events(rolling_buffer, tracking_results)
                BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)
                if len(tracking_results.numpy()) > 0:
                    print(tracking_results.numpy())
        except Exception as e:
            print(f"No Object")
            continue

def landing_prediction(ball_memory,odometry=None):
    from parabola_fitter import ParabolaFitter
    parabola_fitter = ParabolaFitter(camera_rpy_to=[0, 45, -90], camera_rpy_back=[45, 0, 90],
                                     min_data_samples=30)
    params = parabola_fitter.mono_fitter(ball_memory, odometry=odometry)
    x_0, y_0, z_0, vx, vy, vz = params

    impact_t = -x_0 / vx
    impact_point = parabola_fitter.parabola_model_time(params, impact_t)
    print(impact_point)
    pass
def robot_control(landing_queue):
    pass
def signal_handler(sig, frame):
    print("Terminating processes...")
    stop_event.set()
    sys.exit(0)

if __name__ == '__main__':
    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    # Shared objects
    event_queue = Queue(maxsize=5)
    flir_queue = Queue(maxsize=5)
    ROI_queue = Queue(maxsize=5)
    landing_queue = Queue(maxsize=5)
    odom_queue = Queue(maxsize=5)


    stop_event = multiprocessing.Event()

    # Processes
    # cuda_process = Process(target=cuda_test,args=(device,))
    event_process = Process(target=read_from_event, args=(event_queue, stop_event))
    flir_process = Process(target=read_from_flir, args=(flir_queue, stop_event))
    # tracking_process = Process(target=tracking, args=(event_queue, flir_queue, stop_event))
    model_tracking_process = Process(target=model, args=(event_queue, flir_queue, stop_event))

    # Start processes
    # cuda_process.start()
    event_process.start()



    flir_process.start()
    # tracking_process.start()
    model_tracking_process.start()




    # cuda_process.join()
    event_process.join()
    flir_process.join()

    # Wait for processes to finish
    try:
        # cuda_process.join()
        event_process.join()
        flir_process.join()
    except KeyboardInterrupt:
        stop_event.set()
        event_process.terminate()
        flir_process.terminate()
        # tracking_process.terminate()

