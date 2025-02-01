import cv2

import argparse
import os

import matplotlib.pyplot as plt
import sys

import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import multiprocessing
import signal
import numpy as np
import time
from ekf_predict import ekf_predict
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def transform_pixel_to_coordinate(ball_pixels):
    cx = 619.674
    cy = 333.682
    fx = 673.797
    fy = 673.797
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    distort_matrix = np.array([-0.269418, 0.048704, -0.000727, 0.002795, 0])
    distorted_pixel = ball_pixels[:,:2].astype(float)
    # print(distorted_pixel)
    # distorted_pixel = distorted_pixel.astype(float)
    start_t=ball_pixels[0][2]
    # print(start_t)
    time_array=ball_pixels[:,2]-start_t
    time_array = time_array.reshape(-1, 1).astype(float)
    # print(time_array)
    # distorted_pixel = distorted_pixel.reshape(-1, )
    # return intrinsic_matrix, distorted_pixel
    undistorted_pixel = cv2.undistortPoints(distorted_pixel, intrinsic_matrix, distort_matrix, P=intrinsic_matrix).reshape(-1,2)
    # print(undistorted_pixel)

    # return undistorted_pixel
    # undistorted_pixel[:,0] -= cx
    # undistorted_pixel[:,1] -= cy
    # undistorted_pixel[:,0] /= fx
    # undistorted_pixel[:,1] /= fy
    # print(undistorted_pixel,time_array)
    coordinate=np.concatenate((undistorted_pixel,time_array),axis=1)
    # print(coordinate)
    return coordinate

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
def read_from_raw(shared_event_data, stop_event):
    from metavision_core.event_io import EventsIterator

    event_iterator = iter(EventsIterator(input_path="/home/xinchi/Event_RGB_tracking_data/saved_data/event/0.raw",mode="mixed", delta_t=1000, n_events=50000))
    while not stop_event.is_set():
        start_time = time.time()
        try:
            # if event_queue.full():
            #     event_queue.get()  # Remove the oldest item
            event_data = next(event_iterator)
            # event_data[:,[0,1]]=event_data[:,[1,0]]
            event_data['x'],event_data['y']=event_data['y'].copy(),-event_data['x'].copy()
            event_data['y']+=1279
            event_memory,event_dtype,event_shape=shared_event_data
            event_shm = shared_memory.SharedMemory(name=event_memory.name)
            # Attach to the shared memory and create a NumPy array view
            shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
            shm[:] = np.zeros(event_shape, dtype=event_dtype)
            shm[:event_data.shape[0]] = event_data
            time.sleep(0.01)
        except Exception as e:
            # print(f"Error in read_from_event: {e}")
            continue
        end_time = time.time()
        # print("raw fps:", 1 / (end_time - start_time))
def read_from_video(shared_flir_data, stop_event):
    video_path="/home/xinchi/Event_RGB_catching/saved_data_data/rgb/0.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    while not stop_event.is_set():
        try:
            # Read the next frame from the video
            ret, frame = cap.read()

            if not ret:
                # If no frame is returned, the video has ended
                print("End of video.")
                break
            cv_image = frame

            flir_memory,flir_dtype,flir_shape=shared_flir_data
            flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
            shm = np.ndarray(cv_image.shape, dtype=flir_dtype, buffer=flir_shm.buf)
            shm[:] = cv_image
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in read_from_flir: {e}")

def read_from_event(shared_event_data, stop_event):
    from metavision_core.event_io import EventsIterator

    event_iterator = iter(EventsIterator(input_path="",mode="mixed", delta_t=1000, n_events=5000))
    while not stop_event.is_set():
        try:
            # if event_queue.full():
            #     event_queue.get()  # Remove the oldest item
            event_data = next(event_iterator)
            # event_data[:,[0,1]]=event_data[:,[1,0]]
            event_data['x'],event_data['y']=event_data['y'].copy(),-event_data['x'].copy()
            event_data['y']+=1279
            # print(event_data['y'])
            # print(max(event_data['x']), max(event_data['y']))
            # print(event_data.shape)

            # event_queue.put(event_data, timeout=0.01)
            event_memory,event_dtype,event_shape=shared_event_data
            event_shm = shared_memory.SharedMemory(name=event_memory.name)
            # Attach to the shared memory and create a NumPy array view
            shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
            # print(event_data.shape)
            shm[:] = np.zeros(event_shape, dtype=event_dtype)
            shm[:event_data.shape[0]] = event_data
            # print(event_data.shape)
            # print(shm)
            # event_shm.close()
        except Exception as e:
            print(f"Error in read_from_event: {e}")
            continue

def read_from_flir(shared_flir_data, stop_event):
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
            # if flir_queue.full():
            #     flir_queue.get()
            # flir_queue.put(cv_image, timeout=0.1)
            flir_memory,flir_dtype,flir_shape=shared_flir_data
            flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
            shm = np.ndarray(cv_image.shape, dtype=flir_dtype, buffer=flir_shm.buf)
            shm[:] = cv_image
            # print(shm)
            # print(flir_shm)
            # flir_shm.close()
            image_result.Release()
        except Exception as e:
            print(f"Error in read_from_flir: {e}")
            continue


def model(shared_event_data, shared_flir_data,shared_ROI_data, stop_event):
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


    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    first_flir_image = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

    # print(shm)
    # event_shm.close()


    event_template = torch.zeros([1024, 19])
    is_tracker_initialiezed=False
    initial_bbox = None

    # with MTWindow(title="Metavision Events Viewer", width=1080, height=1440,
    #               mode=BaseWindow.RenderMode.BGR) as window:

    # Event Frame Generator
    # event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=1280, sensor_height=720, fps=25,
    #                                                    palette=ColorPalette.Dark)
    #
    # def on_cd_frame_cb(ts, cd_frame):
    #     cd_frame = cv2.rotate(cd_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     # window.show_async(cd_frame)
    #
    # event_frame_gen.set_output_callback(on_cd_frame_cb)


    while not stop_event.is_set():
        start_time = time.time()
        try:
            # event_data = event_queue.get(timeout=0.1)
            # # print(event_data.dtype)
            # flir_data = flir_queue.get(timeout=0.1)
            # event_memory, event_dtype, event_shape = shared_event_data
            # event_shm = shared_memory.SharedMemory(name=event_memory)
            event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
            # print(event_data)
            # flir_memory,flir_dtype,flir_shape=shared_flir_data
            # flir_shm = shared_memory.SharedMemory(name=flir_memory)
            flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

        except Exception as e:
            print(f"No Object")
            continue
        try:

            # event_frame_gen.process_events(event_data)
            # print(event_data.shape)
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
                # if event_data['t'].shape[0]==0:
                #     pass
                # print(event_data['t'].shape)
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
                # if ROI_queue.full():
                #     ROI_queue.get()
                # ROI_queue.put(state,timeout=0.01)
                ROI[0]=state[0]
                ROI[1]=state[1]
                ROI[2]=state[2]
                ROI[3]=state[3]
                # print(state,ROI)
                cv2.rectangle(flir_data, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),color=(0, 255, 0), thickness=2)

            # Display the image
        except Exception as e:
            print(f"No Events")
        cv2.imshow('Image with Bounding Box', flir_data)
        cv2.waitKey(1)
        end_t = time.time()
        # print("model fps:",1/(end_t - start_time))

def pure_event_tracking(shared_event_data,shared_ROI_data,shared_ball_pixel_data,ball_observations,keyboard_input,stop_event):


    from metavision_core.event_io import EventsIterator
    from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
    from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig, draw_tracking_results
    from metavision_sdk_core import BaseFrameGenerationAlgorithm, RollingEventBufferConfig, RollingEventCDBuffer
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
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
            default="",
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
                                         default=50, help='Minimal size of an object to track (in pixels).')
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
    def ROI_filter(evs,ROI,outer_range=5000):

        x,y,w,h = ROI
        # print(ROI)
        # print(f'ROI: {x},{y},{w},{h}')
        x=x*2/3
        y=y*8/9
        # evs=np.array(evs)

        # print(x,max(0,x - outer_range),min(720,x + w + outer_range),y,max(0,y - outer_range),min(1280,y + h + outer_range))
        mask = (
                    (evs['x'] > int(max(0,x - outer_range))) &
                    (evs['x'] < int(min(720,x  + w + outer_range))) &
                    (evs['y'] > int(max(0,y - outer_range))) &
                    (evs['y'] < int(min(1280,y + h + outer_range)))
                )
        # mask= (
        #             (evs['x'] > 300 ) &
        #             (evs['x'] < 400 ) &
        #             (evs['y'] > 500 ) &
        #             (evs['y'] < 600)
        #         )
        new_evs = evs[mask]
        return new_evs



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


    width = 720
    height = 1280
    # Noise + Trail filter that will be applied to events
    activity_noise_filter = ActivityNoiseFilterAlgorithm(width, height, args.activity_time_ths)
    # print("threshold",activity_noise_filter.width,activity_noise_filter.height)
    trail_filter = TrailFilterAlgorithm(width, height, args.activity_trail_ths)

    # Tracking Algorithm
    tracking_config = TrackingConfig()  # Default configuration

    tracking_algo = TrackingAlgorithm(sensor_width=width, sensor_height=height, tracking_config=tracking_config)
    tracking_algo.min_size = args.min_size
    tracking_algo.max_size = args.max_size

    output_img = np.zeros((height, width, 3), np.uint8)

    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    tracking_results = tracking_algo.get_empty_output_buffer()

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)

    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)

    ball_pixel_memory,ball_pixel_dtype,ball_pixel_shape = shared_ball_pixel_data
    ball_pixel_shm = shared_memory.SharedMemory(name=ball_pixel_memory.name)

    ball_pixel = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_shm.buf)
    # Event Frame Generator
    # event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=1080, sensor_height=720, fps=25,
    #                                                    palette=ColorPalette.Dark)

    # def on_cd_frame_cb(ts, cd_frame):
    #     # cd_frame = cv2.rotate(cd_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     # pass
    #     window.show_async(cd_frame)
    #
    # event_frame_gen.set_output_callback(on_cd_frame_cb)


    with MTWindow(title="Generic Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        # window.show_async(output_img)
        while not stop_event.is_set():

            start_t = time.time()
            try:
                raw_evs = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
                ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                # print(ROI)
                # event_frame_gen.process_events(evs)
                # print(raw_evs.shape)
                # print(ROI)
                evs=ROI_filter(raw_evs,ROI)
                # evs=raw_evs
                # print(evs.shape)
                EventLoop.poll_and_dispatch()
                # Process events

                # print(evs.shape)
                activity_noise_filter.process_events(evs, events_buf)
                # print(events_buf.numpy())

                rolling_buffer.insert_events(events_buf.numpy())
                tracking_algo.process_events(rolling_buffer, tracking_results)
                BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer, output_img)

                draw_tracking_results(evs['t'][-1], tracking_results, output_img)
                # output = cv2.rotate(output_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # print(output.shape)
                # cv2.rotate(output_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if keyboard_input.value=='s':
                    # print(tracking_results.numpy())
                    # print(ball_observations)
                    if not len(tracking_results.numpy())==0:
                        ball_num=ball_observations.value
                        ball_pixel[ball_num][0]=tracking_results.numpy()[0][0]
                        ball_pixel[ball_num][1]=tracking_results.numpy()[0][1]
                        ball_pixel[ball_num][2] = time.time()
                        # print(ball_pixel[ball_num][2])
                        ball_pixel[ball_num][3] = tracking_results.numpy()[0][5]
                        ball_pixel[ball_num][4] = tracking_results.numpy()[0][6]
                        # print(time.time())
                        if ball_num<999:
                            ball_num+=1
                        ball_observations.value=ball_num
                    # print(ball_pixel)
                else:
                    ball_observations.value=0
                window.show_async(output_img)

            except Exception as e:
                # print(f"No Object (event)")
                continue
            end_t = time.time()
            # print("event fps:", 1 / (end_t - start_t))


def keyboard(shared_value):
    from pynput import keyboard
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
            shared_value.value = key.char if hasattr(key, 'char') else str(key)
        except AttributeError:
            shared_value.value = str(key)

    with keyboard.Listener(on_press=on_press) as listener:

        listener.join()
def landing_prediction(shared_ball_pixel_data,ball_observations,shared_impact_point_data,stop_event,odometry=None):
    from parabola_fitter import ParabolaFitter

    ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape = shared_ball_pixel_data
    ball_pixel_shm = shared_memory.SharedMemory(name=ball_pixel_memory.name)


    # print(ball_pixels)

    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)

    parabola_fitter = ParabolaFitter(camera_rpy_to=[0, 45, -90], camera_rpy_back=[45, 0, 90],
                                     min_data_samples=30)
    while not stop_event.is_set():
        ball_pixels = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_shm.buf)
        ball_pixels = ball_pixels[:ball_observations.value]
        # print(len(ball_pixels))
        try:


            ball_memory = transform_pixel_to_coordinate(ball_pixels)
            if len(ball_memory)>=500:
                print("ball_pixels", ball_memory)
                np.save("ball_pixels", ball_memory)
                break
            if len(ball_memory)>=500:
                print("ball_memory", ball_memory)
                np.save("ball_memory", ball_memory)
                break
            # print(ball_memory.shape)
            # params = parabola_fitter.mono_fitter(ball_memory, odometry=odometry)
            # # print(params)
            # x_0, y_0, z_0, vx, vy, vz = params
            # impact_t = -x_0 / vx
            x,y,z = ekf_predict(ball_memory)

            impact_point_array[0]=x
            impact_point_array[1]=y
            impact_point_array[2]=z
            # print(impact_point_array.shape)
            # print(x,y,z)
            # print(impact_point_array)
        except:
            continue



def robot_control(shared_impact_point_data,keyboard_input,stop_event):
    from pcan_cybergear import CANMotorController
    import can
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
    motor1 = CANMotorController(bus, motor_id=101, main_can_id=254)
    motor1.set_0_pos()
    motor1.enable()
    while not stop_event.is_set():

        # print(len(ball_pixels))
        try:
            if keyboard_input.value=='s':
                impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)

                # print(impact_point_array.shape)
                x, y, z = impact_point_array
                # write your code here
                # print(x,y,z)
                shaft_length = 0.5
                joint_height = 0.7
                delta_height = z + joint_height
                if delta_height>0 and delta_height<0.5:
                    angle = np.pi/2 - np.arctan(delta_height/shaft_length)
                    # if angle<0:
                    #     angle = max(angle, np.pi/4)
                    if angle>0:
                        angle=min(angle, np.pi/4)
                    if y < 0:
                        angle = -angle
                    motor1.send_motor_control_command(torque=0, target_angle=angle, target_velocity=0, Kp=100, Kd=1)
                    time.sleep(2.5)
                    motor1.disable()
                    bus.shutdown()
        except:
            continue
def signal_handler(sig, frame):
    print("Terminating processes...")
    stop_event.set()
    sys.exit(0)

if __name__ == '__main__':
    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    # Shared objects
    # event_queue = Queue(maxsize=5)
    # flir_queue = Queue(maxsize=5)
    # ROI_queue = Queue(maxsize=5)
    # object_queue = Queue(maxsize=5)
    # landing_queue = Queue(maxsize=5)
    # odom_queue = Queue(maxsize=5)



    # flir_array=np.zeros((1080,1440,3),dtype=dtype)
    # ROI_array=np.zeros((4),dtype=dtype)
    # object_array=np.zeros((4),dtype=dtype)
    # landing_array=np.zeros((4),dtype=dtype)
    # odom_array=np.zeros((4),dtype=dtype)



    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape=(5000,)
    event_memory=shared_memory.SharedMemory(create=True,size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data=(event_memory,event_dtype,event_shape)

    flir_dtype=np.uint8
    flir_shape=(1440,1080,3)
    flir_memory = shared_memory.SharedMemory(create=True, size=np.prod(flir_shape) * np.dtype(flir_dtype).itemsize)
    shared_flir_data=(flir_memory,flir_dtype,flir_shape)
    stop_event = multiprocessing.Event()

    ROI_dtype=np.uint16
    ROI_shape=(4,)
    ROI_memory=shared_memory.SharedMemory(create=True, size=np.prod(ROI_shape) * np.dtype(ROI_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_memory.buf)
    ROI[:]=np.zeros(4)
    shared_ROI_data=(ROI_memory,ROI_dtype,ROI_shape)

    ball_pixel_dtype = np.float64
    ball_pixel_shape = (1000,5)
    ball_pixel_memory = shared_memory.SharedMemory(create=True, size=np.prod(ball_pixel_shape) * np.dtype(ball_pixel_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ball_pixel = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_memory.buf)
    ball_pixel[:] = np.zeros(ball_pixel_shape)
    shared_ball_pixel_data = (ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape)

    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True, size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    impact_point= np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(3)
    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    keyboard_input = Value('u', 's')
    ball_observations = Value('i',0)
    # Processes
    event_process = Process(target=read_from_raw, args=(shared_event_data, stop_event))
    flir_process = Process(target=read_from_video, args=(shared_flir_data, stop_event))
    model_tracking_process = Process(target=model, args=(shared_event_data,shared_flir_data,shared_ROI_data, stop_event))
    event_tracking_process = Process(target=pure_event_tracking,args=(shared_event_data,shared_ROI_data,shared_ball_pixel_data,ball_observations,keyboard_input, stop_event))
    landing_prediction_process = Process(target=landing_prediction, args=(shared_ball_pixel_data,ball_observations,shared_impact_point_data,stop_event))
    keyboard_process = Process(target=keyboard, args=(keyboard_input,))
    # robot_control_process = Process(target=robot_control, args=(shared_impact_point_data,keyboard_input,stop_event))

    # Start processes
    event_process.start()
    flir_process.start()
    # model_tracking_process.start()
    event_tracking_process.start()
    landing_prediction_process.start()
    keyboard_process.start()
    # robot_control_process.start()

    # Wait for processes to finish
    try:
        # cuda_process.join()
        event_process.join()
        flir_process.join()
        # model_tracking_process.join()
        event_tracking_process.join()
        landing_prediction_process.join()
        keyboard_process.join()
        # robot_control_process.join()
    except KeyboardInterrupt:
        stop_event.set()
        event_process.terminate()
        flir_process.terminate()
        model_tracking_process.terminate()
        event_tracking_process.terminate()
        landing_prediction_process.terminate()
        keyboard_process.terminate()
        # robot_control_process.terminate()

