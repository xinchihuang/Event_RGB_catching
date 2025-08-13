import math
import cv2
import os
import sys
import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import multiprocessing
import signal
import numpy as np
import time
import argparse
from gi.overrides.Gtk import initialized
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from event_process import *
from RGB_process import *
from fitting_process import *
def starter(system_state,stop_event):
    time.sleep(3)
    print("Starting system...")
    system_state.value = 't'
def signal_handler(sig, frame):
    print("Terminating processes...")
    stop_event.set()
    sys.exit(0)
def keyboard(shared_value,system_state):
    from pynput import keyboard
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
            # shared_value.value = key.char if hasattr(key, 'char') else str(key)
            if key.char=='t':
                system_state.value='t'
        except AttributeError:
            system_state.value='e'

    with keyboard.Listener(on_press=on_press) as listener:

        listener.join()
def reseter_for_multi_record(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,
                             shared_impact_point_data,raw_count,num_event,record_index,start_ts,event_broadcasting,video_broadcasting,system_state,stop_event):
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
        # print(event_broadcasting.value,video_broadcasting.value)

        if event_broadcasting.value=='f' and video_broadcasting.value=='f':
            system_state.value='r'
        # if event_broadcasting.value=='f' :
        #     system_state.value='r'
        if system_state.value == 'r':
            print("Waiting for reset")
            time.sleep(2)
            system_state.value = 'w'
            print("Set to Waiting")
            time.sleep(2)
            print("Cleaning memory...")
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

            raw_count.value=0
            # 重置 num_event
            num_event.value = 0
            # 下一个record

            start_ts.value = 0.0
            video_broadcasting.value='s'
            event_broadcasting.value='s'
            print("Restarting system...")
            time.sleep(2)
            record_index.value += 1
            system_state.value = 'e'

def check_align(shared_event_data,shared_flir_data,num_event,system_state,stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=0.0,
                                                       palette=ColorPalette.Dark)

    print(event_frame_gen.get_accumulation_time_us())
    print(event_frame_gen.get_fps())
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    dbscan = DBSCAN(eps=9, min_samples=70)
    count=0
    # time.sleep(1)
    # root_dir = "/home/xinchi/Event_RGB_catching_data/dataset"
    # index=len([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    # if not os.path.exists(f"{root_dir}/{index}"):
    #     os.makedirs(f"{root_dir}/{index}")
    #     os.makedirs(f"{root_dir}/{index}/combined")
    #     os.makedirs(f"{root_dir}/{index}/rgb")
    #     os.makedirs(f"{root_dir}/{index}/event")
    # time.sleep(10)
    # print("start tracking")
    # system_state.value='t'
    buffered_frame=np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    while not stop_event.is_set():
        if system_state.value == 'e':
            break
        # start_time = time.time()
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
        event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
        if num_event.value==0:
            continue
        event_data= event_data[:num_event.value]
        rgb_image=flir_data
        event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame,accumulation_time_us=1000)
        image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
        dbscan.fit(points)
        labels = dbscan.labels_
        event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cluster_points = points[labels != -1]
        if not len(cluster_points) == 0:
            y = min(cluster_points[:, 0])
            x = min(cluster_points[:, 1])
            w = max(cluster_points[:, 0]) - y
            h = max(cluster_points[:, 1]) - x
            # cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
            # cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
        combined = np.hstack((rgb_image, event_image))
        scale = 0.8
        # 获取新的尺寸
        width = int(combined.shape[1] * scale)
        height = int(combined.shape[0] * scale)
        new_size = (width, height)

        resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

        cv2.imshow('Image with Bounding Box', resized_img)
        cv2.waitKey(1)
        # cv2.imwrite(f"{root_dir}/{0}/combined/" + str(count) + ".jpg", combined)
        # cv2.imwrite(f"{root_dir}/{index}/event/" + str(rgb_index) + ".jpg", event_image)

        count += 1
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Event RGB Catching System')
    parser.add_argument('--record_index', type=int, default=10,
                       help='Initial record index (default: 40)')
    args = parser.parse_args()
    
    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    # Shared objects
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape=(100000,)
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
    event_broadcasting = Value('u', 's')
    video_broadcasting = Value('u', 's')
    ball_observations = Value('i',0)
    raw_count = Value('i', 0)
    num_event = Value('i', 0)
    record_index = Value('i', args.record_index)
    start_ts = Value('d', 0.0)
    
    print(f"Starting with record_index: {args.record_index}")

    record_dir='/media/xinchi/data/Event_RGB_catching_data/saved_data_dc'

    # Processes
    event_process = Process(target=read_from_raw, args=(shared_event_data,system_state,num_event,raw_count,record_index,start_ts,stop_event,event_broadcasting,record_dir))
    flir_process = Process(target=read_from_video, args=(shared_flir_data,system_state,raw_count,record_index,start_ts, stop_event,video_broadcasting,record_dir))
    model_tracking_process = Process(target=model, args=(shared_event_data,shared_flir_data,shared_ROI_data,num_event,system_state, stop_event))
    event_tracking_process = Process(target=event_tracking_3D,args=(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_event,system_state, stop_event))
    RGB_tracking_process = Process(target=RGB_tracking_3D,
                                     args=(shared_event_data, shared_flir_data, shared_ROI_data,
                                           shared_ball_pixel_3D_data, ball_observations, num_event, system_state,
                                           stop_event))
    landing_prediction_process = Process(target=landing_prediction, args=(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,record_index,system_state,stop_event))
    keyboard_process = Process(target=keyboard, args=(keyboard_input,system_state))
    robot_control_process = Process(target=robot_control, args=(shared_impact_point_data,system_state,stop_event))
    checker_process=Process(target=check_align,args=(shared_event_data,shared_flir_data,num_event,system_state,stop_event))
    reset_process= Process(target=reseter_for_multi_record, args=(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,
                             shared_impact_point_data,raw_count,num_event,record_index,start_ts,event_broadcasting,video_broadcasting,system_state,stop_event))
    starter_process = Process(target=starter, args=(system_state,stop_event))

    # Start processes
    # starter_process.start()
    event_process.start()
    flir_process.start()
    # checker_process.start()
    model_tracking_process.start()
    event_tracking_process.start()
    # RGB_tracking_process.start()
    landing_prediction_process.start()
    reset_process.start()
    # keyboard_process.start()
    # robot_control_process.start()
    # Wait for processes to finish or system_state becomes 'e'
    try:
        while True:
            # Check if system should exit
            if system_state.value == 'e':
                print("System exit requested, terminating processes...")
                stop_event.set()
                break
            
            # Check if all active processes are still alive
            active_processes = [starter_process, event_process, flir_process, event_tracking_process, landing_prediction_process, reset_process]
            if not any(p.is_alive() for p in active_processes):
                print("All processes finished naturally")
                break
            
            time.sleep(0.1)  # Check every 100ms
        
        # Give processes time to clean shutdown
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
        stop_event.set()
        time.sleep(2)
        # event_process.terminate()
        # flir_process.terminate()
        # model_tracking_process.terminate()
        # event_tracking_process.terminate()
        # landing_prediction_process.terminate()
        # keyboard_process.terminate()
        # robot_control_process.terminate()
    finally:
        print("Cleaning up shared memory...")
        for shm in [event_memory, flir_memory, ROI_memory, ball_pixel_memory,
                    ball_pixel_3D_memory, impact_point_memory]:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # Already unlinked
                pass

