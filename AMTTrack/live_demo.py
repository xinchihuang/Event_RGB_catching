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
# from ekf_predict import ekf_predict
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
def ROI_filter(events, ROI_data):
    """
    Return the sub-array of `events` whose (x, y) lie inside a rectangular ROI.

    Parameters
    ----------
    events : np.ndarray
        Structured array with the dtype shown above.  Shape: (N,)
    ROI_data : tuple (x_r, y_r, w_r, h_r)
        Top-left corner (x_r, y_r) and width/height of the rectangle.

    Returns
    -------
    np.ndarray
        A **view** (no copy) of the original array containing only the
        events inside the ROI.  Use .copy() if you need an independent copy.
    """
    x_r, y_r, w_r, h_r = ROI_data
    events_cp=events.copy()
    mask = (
            (events_cp['x'] >= x_r) & (events_cp['x'] < x_r + w_r) &
            (events_cp['y'] >= y_r) & (events_cp['y'] < y_r + h_r)
    )

    new_event=events_cp[mask]
    return new_event
def align_to_rgb(event_data, rgb_width=1080, rgb_height=1440, event_width=720, event_height=1280):
    x_e,y_e=event_data['y'].copy(),-event_data['x'].copy()+1279
    # print(x_e,y_e)
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2
    event_pixel = 3.6
    rgb_pixel = 4.86
    x=(x_e-event_center_x)*rgb_pixel/event_pixel+rgb_center_x
    y=(y_e-event_center_y)*rgb_pixel/event_pixel+rgb_center_y
    # print(x,y)
    return x.astype(int),y.astype(int)
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

def keyboard(shared_value,system_state):
    from pynput import keyboard
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
            # shared_value.value = key.char if hasattr(key, 'char') else str(key)
            if key.char in ('e', 't', 'c','w'):
                system_state.value=key.char
        except AttributeError:
            system_state.value='e'

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
def read_from_event(shared_event_data,system_state, stop_event):
    from metavision_core.event_io import EventsIterator
    event_iterator = iter(EventsIterator(input_path="",mode="mixed", delta_t=10000, n_events=10000))
    print("start event camera")
    while system_state.value == 'w':
        continue
    while not stop_event.is_set():
        event_data = next(event_iterator)
        if system_state.value == 'e':
            break
        if system_state.value == 't':
            try:
                event_data['x'],event_data['y']=event_data['y'].copy(),-event_data['x'].copy()
                event_data['y']+=1279
                event_memory,event_dtype,event_shape=shared_event_data
                event_shm = shared_memory.SharedMemory(name=event_memory.name)
                # Attach to the shared memory and create a NumPy array view
                shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
                # print(event_data.shape)
                shm[:] = np.zeros(event_shape, dtype=event_dtype)
                shm[:event_data.shape[0]] = event_data
            except Exception as e:
                print(f"Error in read_from_event: {e}")
                continue

def read_from_flir(shared_flir_data,system_state, stop_event):
    import PySpin
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()
    cam.BeginAcquisition()
    print("start rgb camera")
    while system_state.value == 'w':
        continue
    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        if system_state.value == 't':
            try:
                image_result = cam.GetNextImage(10000)
                if image_result.IsIncomplete():
                    continue
                image_data = image_result.GetNDArray()
                cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
                cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
                flir_memory,flir_dtype,flir_shape=shared_flir_data
                flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
                shm = np.ndarray(cv_image.shape, dtype=flir_dtype, buffer=flir_shm.buf)
                shm[:] = cv_image
                image_result.Release()
            except Exception as e:
                print(f"Error in read_from_flir: {e}")
                continue
def event_tracking_3D(shared_event_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,system_state, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_width, event_height = 720, 1280
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_width, sensor_height=event_height,
                                                       fps=100,
                                                       palette=ColorPalette.Dark)

    ball_pixel_3D_memory,ball_pixel_3D_dtype,ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    save_dir = "/home/xinchi/Event_RGB_catching_data/frames"
    num_file = len(
        [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])

    event_dir = f"{save_dir}/{num_file}/event/"
    if not os.path.exists(event_dir):
        os.makedirs(event_dir)  # Corrected function name
    dbscan = DBSCAN(eps=10, min_samples=80)
    count=0
    print("start event tracking")
    while system_state.value == 'w':
        continue

    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # print(system_state.value)
        if system_state.value == 't':
            # start_t=time.time()
            try:
                raw_event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
                # ROI_data = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                ROI_data = [0, 0, 720, 800]
                event_data = ROI_filter(raw_event_data, ROI_data)
                # print(max(event_data['y']))
                t = event_data['t'][-1] / 1000000
                # # print(t)
                event_frame = np.zeros((event_height, event_width, 3), dtype=np.uint8)
                event_frame = np.ascontiguousarray(event_frame)
                event_frame_gen.generate_frame(event_data, event_frame)
                image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
                points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
                if points.shape[0] == 0:
                    continue
                dbscan.fit(points)
                labels = dbscan.labels_
                color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cluster_points = points[labels != -1]
                if not len(cluster_points)==0:
                    y = min(cluster_points[:, 0])
                    x = min(cluster_points[:, 1])
                    w = max(cluster_points[:, 0]) - y
                    h = max(cluster_points[:, 1]) - x
                    cv2.rectangle(color_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
                    ball_num = ball_observations.value
                    ball_pixel_3D[ball_num][0] = float(x + h / 2)
                    ball_pixel_3D[ball_num][1] = float(y + h / 2)
                    ball_pixel_3D[ball_num][2] = float(h)
                    ball_pixel_3D[ball_num][3] = t
                    if ball_num < 100:
                        ball_num += 1
                    ball_observations.value = ball_num
                cv2.imshow('event ', color_image)
                cv2.waitKey(1)
                # print("iters:",count)
                # print("ball_observations",ball_observations.value)
                count += 1
                end_t = time.time()
                # print("Event time:", end_t - start_t)
            except Exception as e:
                # print(f"Error in event tracking: {e}")
                continue

def landing_prediction(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,system_state,stop_event):

    from parabola_fitter import MonoRansac
    from utils import rotate_matrix_coordinate

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

    ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape = shared_ball_pixel_3D_data
    ball_pixel_shm = shared_memory.SharedMemory(name=ball_pixel_memory.name)
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
    fitter = MonoRansac(camera_params=camera_config,rotation_matrix_to=rotation_matrix_to,rotation_matrix_back=rotation_matrix_back)
    print("start landing prediction")
    while system_state.value == 'w':
        continue
    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        if system_state.value == 't':
            try:
                ball_pixels = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_shm.buf)
                observations = ball_pixels[:ball_observations.value]
                if len(observations) >= 10:
                    observations[:, 3] = observations[:, 3] - observations[0, 3]
                    fitter.set_data(observations)
                    result = fitter.get_params()
                    impact_point=fitter.predict()
                    impact_point_array[0] = impact_point[0]
                    impact_point_array[1] = impact_point[1]
                    impact_point_array[2] = impact_point[2]
                    system_state.value='c'
                    # print(ball_pixels)
                    # ball_pixels[:] = np.zeros(ball_pixel_shape)
                    # ball_observations.value = 0

            except Exception as e:
                    print(f"Error in Landing prediction: {e}")
                    continue

def robot_control(shared_impact_point_data,system_state,stop_event):
    from pcan_cybergear import CANMotorController
    import can
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
    motor1 = CANMotorController(bus, motor_id=101, main_can_id=254)
    motor1.set_0_pos()
    motor1.enable()
    motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0, Kp=100, Kd=1)
    print("motor start")
    while system_state.value == 'w':
        continue
    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        if system_state.value == 'c':
            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            # print(impact_point_array)
            x, y, z = impact_point_array
            shaft_length = 0.5
            joint_height = 0.45
            delta_height = z - joint_height
            if delta_height>0:
                angle = np.pi/2 - np.arctan(delta_height/shaft_length)
                if angle>0:
                    angle=min(angle, np.pi/4)
                if y < 0:
                    angle = -angle
                print('angle',angle)
                motor1.send_motor_control_command(torque=0, target_angle=angle, target_velocity=0, Kp=100, Kd=1)
                time.sleep(2)
                print('reset')
                for degree in range(100):
                    motor1.send_motor_control_command(torque=0, target_angle=angle*(1-degree/100), target_velocity=0, Kp=10, Kd=1)
                    time.sleep(0.05)
                motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0,Kp=100, Kd=5)
                time.sleep(0.05)
                # motor1.disable()
                # bus.shutdown()
            system_state.value='r'

def reseter(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,system_state,stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)

    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        if system_state.value == 'r':
            # 清空 event memory
            event_array = np.ndarray(event_shape, dtype=np.dtype(event_dtype), buffer=event_shm.buf)
            event_array[:] = np.zeros(event_shape, dtype=np.dtype(event_dtype))

            # 清空 flir image
            flir_array = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
            flir_array[:] = 0

            # 清空 3D ball pixel
            ball_pixel_3D_array = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype,
                                             buffer=ball_pixel_3D_shm.buf)
            ball_pixel_3D_array[:] = 0

            # 清空 ROI
            ROI_array = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
            ROI_array[:] = 0

            # 清空 impact point
            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            impact_point_array[:] = 0

            # 重置 ball_observations
            ball_observations.value = 0

            # 重置 system_state
            system_state.value='t'
            time.sleep(0.01)
            print("reset system state to tracking")




def signal_handler(sig, frame):
    print("Terminating processes...")
    stop_event.set()
    sys.exit(0)

def checker(shared_event_data,shared_flir_data,system_state,stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_width, event_height = 720, 1280
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_width, sensor_height=event_height,
                                                       fps=100,
                                                       palette=ColorPalette.Dark)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    save_dir = "/home/xinchi/Event_RGB_catching_data/demo_data"
    num_file = len(
        [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])
    event_dir = f"{save_dir}/{num_file}/event/"
    rgb_dir = f"{save_dir}/{num_file}/rgb/"
    if not os.path.exists(event_dir):
        os.makedirs(event_dir)
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    count=0
    while not stop_event.is_set():
        event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
        event_frame = np.zeros((event_height,event_width,3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data,event_frame)
        image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        event_image=cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # print("frame:", count)

        # cv2.imshow('Event ', event_image)
        # cv2.imshow('rgb ', flir_data)
        # cv2.waitKey(1)
        if system_state.value == 's':
            rgb_path = f"{rgb_dir}/{count}.jpg"
            event_path = f"{event_dir}/{count}.jpg"
            cv2.imwrite(event_path, event_image)
            cv2.imwrite(rgb_path,flir_data)

        count += 1


if __name__ == '__main__':
    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape=(10000,)
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
    ball_pixel_shape = (100,3)
    ball_pixel_memory = shared_memory.SharedMemory(create=True, size=np.prod(ball_pixel_shape) * np.dtype(ball_pixel_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ball_pixel = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_memory.buf)
    ball_pixel[:] = np.zeros(ball_pixel_shape)
    shared_ball_pixel_data = (ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape)

    ball_pixel_3D_dtype = np.float64
    ball_pixel_3D_shape = (1000, 4)
    ball_pixel_3D_memory = shared_memory.SharedMemory(create=True,
                                                   size=np.prod(ball_pixel_3D_shape) * np.dtype(ball_pixel_3D_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_memory.buf)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    shared_ball_pixel_3D_data = (ball_pixel_3D_memory, ball_pixel_3D_dtype, ball_pixel_3D_shape)

    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(create=True, size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize)
    # ROI_memory=np.zeros(4)
    impact_point= np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_memory.buf)
    impact_point[:] = np.zeros(3)
    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    keyboard_input = Value('u', 'e')
    system_state=Value('u', 'w')
    ball_observations = Value('i',0)
    # Processes
    event_process = Process(target=read_from_event, args=(shared_event_data,system_state, stop_event))
    flir_process = Process(target=read_from_flir, args=(shared_flir_data,system_state, stop_event))
    event_tracking_process = Process(target=event_tracking_3D,args=(shared_event_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,system_state, stop_event))
    landing_prediction_process = Process(target=landing_prediction, args=(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,system_state,stop_event))
    keyboard_process = Process(target=keyboard, args=(keyboard_input,system_state))
    robot_control_process = Process(target=robot_control, args=(shared_impact_point_data,system_state,stop_event))
    checker_process = Process(target=checker, args=(shared_event_data,shared_flir_data,system_state,stop_event))
    reset_process=Process(target=reseter,args=(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,system_state,stop_event))

    # Start processes
    keyboard_process.start()
    event_process.start()
    flir_process.start()
    event_tracking_process.start()
    landing_prediction_process.start()
    robot_control_process.start()
    reset_process.start()
    # checker_process.start()

    # Wait for processes to finish
    try:
        event_process.join()
        flir_process.join()
        event_tracking_process.join()
        landing_prediction_process.join()
        keyboard_process.join()
        robot_control_process.join()
    except KeyboardInterrupt:
        stop_event.set()
        event_process.terminate()
        flir_process.terminate()
        event_tracking_process.terminate()
        landing_prediction_process.terminate()
        keyboard_process.terminate()
        robot_control_process.terminate()

