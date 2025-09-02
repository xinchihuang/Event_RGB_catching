import cv2
import pyttsx3
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

# from torch.fx.experimental import recording

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def read_from_raw(shared_event_data, stop_event):
    from metavision_core.event_io import EventsIterator

    event_iterator = iter(EventsIterator(input_path="/home/xinchi/Event_RGB_catching_data/saved_data/event/5.raw",mode="mixed", delta_t=1000, n_events=5000))
    while not stop_event.is_set():
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
        except Exception as e:
            # print(f"Error in read_from_event: {e}")
            continue
def read_from_video(shared_flir_data, stop_event):
    video_path="/home/xinchi/Event_RGB_catching_data/saved_data/rgb/0.mp4"
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

def record_event(shared_event_data, keyboard_input, stop_event, sync_timestamps):
    from metavision_core.event_io import EventsIterator
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
    import os

    start_time = time.time()

    event_width, event_height = 720, 1280

    device = initiate_device("")
    save_dir="/home/xinchi/Event_RGB_catching_data/saved_data"
    recording = False
    mv_iterator = EventsIterator.from_device(device=device)
    event_iterator = iter(EventsIterator.from_device(device=device))
    # event_width, event_height = event_iterator.get_size()  # Camera Geometry


    # Event频率控制: 500Hz (10倍于RGB的50Hz)
    event_fps = 500
    event_interval = 1.0 / event_fps  # 0.002s per event frame
    
    with MTWindow(title="Metavision Events Viewer", width=event_width, height=event_height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_height, sensor_height=event_width, fps=event_fps,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            cd_frame = cv2.rotate(cd_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)
        
        last_event_time = None
        event_frame_count = 0
        recording_start_time = None

        while not stop_event.is_set():
            current_time = time.time()
            
            # 精确的Event频率控制
            if last_event_time is not None:
                time_since_last = current_time - last_event_time
                if time_since_last < event_interval:
                    sleep_time = event_interval - time_since_last
                    time.sleep(sleep_time)
                    current_time = time.time()

            key = keyboard_input.value

            if key == 'r' and recording == False:
                recording = True
                recording_start_time = current_time
                # 记录同步时间戳
                sync_timestamps.append(('event_start', recording_start_time))
                print(f"Event recording started at: {recording_start_time:.6f}")
                
                out_path = os.path.join(save_dir, "event")
                num_file = len([f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))])
                log_path = f"{out_path}/{num_file}.raw"

                if device.get_i_events_stream():
                    print(f'Recording to {log_path}')
                    device.get_i_events_stream().log_raw_data(log_path)
                else:
                    print("Event recording already initialized.")

            if key == 'f' and recording == True:
                recording = False
                recording_end_time = current_time
                # 记录同步时间戳
                sync_timestamps.append(('event_end', recording_end_time))
                print(f"Event recording stopped at: {recording_end_time:.6f}")

                if device.get_i_events_stream():
                    device.get_i_events_stream().stop_log_raw_data()
                    
                if recording_start_time:
                    actual_duration = recording_end_time - recording_start_time
                    actual_fps = event_frame_count / actual_duration if actual_duration > 0 else 0
                    print(f"Event: {event_frame_count} frames, {actual_duration:.3f}s, actual FPS: {actual_fps:.2f}")
                    event_frame_count = 0

            try:
                event_data = next(event_iterator)
                EventLoop.poll_and_dispatch()
                event_frame_gen.process_events(event_data)
                
                if recording:
                    event_frame_count += 1
                    
            except StopIteration:
                continue
            
            last_event_time = current_time
            # # Process event data
            # if event_data is not None:
            #     EventLoop.poll_and_dispatch()
            #     event_frame_gen.process_events(event_data)


def record_flir(shared_flir_data, keyboard_input, stop_event, sync_timestamps):
    import PySpin
    import os
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list[0]
    cam.Init()
    cam.BeginAcquisition()

    start_time = time.time()
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    first_flir_image = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
    
    rgb_width, rgb_height = 1080, 1440
    rgb_frame_rate = 50  # 50Hz RGB
    rgb_frame_interval = 1.0 / rgb_frame_rate  # 0.02s per frame
    rgb_frame_size = (rgb_width, rgb_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_video = None
    recording = False
    save_dir = "/home/xinchi/Event_RGB_catching_data/saved_data"
    print(f"Press r to start recording, f to stop recording, and q to quit.")
    print(f"RGB Frame Rate: {rgb_frame_rate}Hz (interval: {rgb_frame_interval:.3f}s)")
    
    frame_count = 0
    recording_start_time = None
    last_frame_time = None
    
    while not stop_event.is_set():
        current_time = time.time()
        
        # 精确的帧率控制
        if last_frame_time is not None:
            time_since_last = current_time - last_frame_time
            if time_since_last < rgb_frame_interval:
                sleep_time = rgb_frame_interval - time_since_last
                time.sleep(sleep_time)
                current_time = time.time()
        
        image_result = cam.GetNextImage(1000)  # 减少超时时间
        if image_result.IsIncomplete():
            continue
            
        image_data = image_result.GetNDArray()
        cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        flir_data = cv_image
        
        key = keyboard_input.value
        if key == 'r' and recording == False:
            recording = True
            recording_start_time = current_time
            # 记录同步时间戳
            sync_timestamps.append(('rgb_start', recording_start_time))
            print(f"RGB recording started at: {recording_start_time:.6f}")
            
            if out_video is None:
                print("Initializing video recording...")
                out_path = os.path.join(save_dir, "rgb")
                num_file = len([f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))])
                output_filename = os.path.join(out_path, f"{num_file}.mp4")
                out_video = cv2.VideoWriter(output_filename, fourcc, rgb_frame_rate, rgb_frame_size)

        if key == 'f' and recording == True:
            recording = False
            recording_end_time = current_time
            # 记录同步时间戳
            sync_timestamps.append(('rgb_end', recording_end_time))
            print(f"RGB recording stopped at: {recording_end_time:.6f}")
            
            if out_video:
                out_video.release()
                out_video = None
                actual_duration = recording_end_time - recording_start_time
                actual_fps = frame_count / actual_duration if actual_duration > 0 else 0
                print(f"RGB: {frame_count} frames, {actual_duration:.3f}s, actual FPS: {actual_fps:.2f}")
                frame_count = 0
        
        # 只在录制时计算帧
        if recording:
            frame_count += 1
            
        # 处理和显示图像
        if flir_data is not None:
            if (flir_data.shape[1], flir_data.shape[0]) != rgb_frame_size:
                flir_data = cv2.resize(flir_data, rgb_frame_size)
            
            scale = 0.6
            width = int(flir_data.shape[1] * scale)
            height = int(flir_data.shape[0] * scale)
            resized_img = cv2.resize(flir_data, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow('flir', resized_img)
            
            # 严格按帧率写入视频
            if recording and out_video:
                out_video.write(flir_data)
        
        last_frame_time = current_time
        # Handle key events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # Cleanup resources
    if recording and out_video:
        out_video.release()
    cv2.destroyAllWindows()
def keyboard(shared_value):
    from pynput import keyboard
    engine = pyttsx3.init()
    engine.say("system start event")
    engine.runAndWait()
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            save_dir = "/home/xinchi/Event_RGB_catching_data/saved_data"
            print(f"Key {key} pressed")
            out_path = os.path.join(save_dir, "event")
            num_file = len(
                [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))])
            if key.char=='r':
                shared_value.value = key.char if hasattr(key, 'char') else str(key)
                engine.say(f"{num_file}")
                engine.runAndWait()
                engine.say(f"throw the ball now")
                engine.runAndWait()

            elif key.char=='f':
                shared_value.value = key.char if hasattr(key, 'char') else str(key)
                engine.say(f"finish recording")
                engine.runAndWait()
                engine.runAndWait()
            elif key.char=='q':
                shared_value.value = key.char if hasattr(key, 'char') else str(key)
        except AttributeError:
            shared_value.value = str(key)

    with keyboard.Listener(on_press=on_press) as listener:

        listener.join()
def show(shared_event_data, shared_flir_data, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_width, event_height = 720, 1280
    with MTWindow(title="Metavision Events Viewer", width=event_width, height=event_height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_width, sensor_height=event_height,
                                                           fps=100,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        while not stop_event.is_set():
            event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
            flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(event_data)
            # cv2.imshow('Image ', flir_data)
            # cv2.waitKey(1)

def signal_handler(sig, frame):
    print("Terminating processes...")
    stop_event.set()
    sys.exit(0)

if __name__ == '__main__':
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
    event_shape=(5000,)
    event_memory=shared_memory.SharedMemory(create=True,size=np.dtype(event_dtype).itemsize * event_shape[0])
    shared_event_data=(event_memory,event_dtype,event_shape)

    flir_dtype=np.uint8
    flir_shape=(1440,1080,3)
    flir_memory = shared_memory.SharedMemory(create=True, size=np.prod(flir_shape) * np.dtype(flir_dtype).itemsize)
    shared_flir_data=(flir_memory,flir_dtype,flir_shape)
    stop_event = multiprocessing.Event()

    keyboard_input = Value('u', 's')
    
    # 添加同步时间戳管理器
    manager = Manager()
    sync_timestamps = manager.list()

    record=False
    if record==True:
        record_flir_process = Process(target=record_flir, args=(shared_flir_data, keyboard_input, stop_event, sync_timestamps))
        record_event_process = Process(target=record_event, args=(shared_event_data, keyboard_input, stop_event, sync_timestamps))
        keyborad_process = Process(target=keyboard, args=(keyboard_input,))

        record_flir_process.start()
        record_event_process.start()
        keyborad_process.start()

        # Wait for processes to finish
        try:

            record_flir_process.join()
            record_event_process.join()
            keyborad_process.join()

        except KeyboardInterrupt:
            stop_event.set()
            record_flir_process.terminate()
            record_event_process.terminate()
            keyborad_process.terminate()
        
        # 打印同步时间戳信息
        if sync_timestamps:
            print("\n=== 同步时间戳记录 ===")
            for event_type, timestamp in sync_timestamps:
                print(f"{event_type}: {timestamp:.6f}")
            
            # 计算RGB和Event的时间差
            rgb_start = next((ts for event, ts in sync_timestamps if event == 'rgb_start'), None)
            event_start = next((ts for event, ts in sync_timestamps if event == 'event_start'), None)
            rgb_end = next((ts for event, ts in sync_timestamps if event == 'rgb_end'), None)
            event_end = next((ts for event, ts in sync_timestamps if event == 'event_end'), None)
            
            if rgb_start and event_start:
                start_diff = abs(rgb_start - event_start)
                print(f"开始时间差: {start_diff*1000:.3f}ms")
            
            if rgb_end and event_end:
                end_diff = abs(rgb_end - event_end)
                print(f"结束时间差: {end_diff*1000:.3f}ms")
    else:
        # Processes
        event_process = Process(target=read_from_raw, args=(shared_event_data, stop_event))
        flir_process = Process(target=read_from_video, args=(shared_flir_data, stop_event))
        show_process = Process(target=show, args=(shared_event_data, shared_flir_data, stop_event))

        # Start processes
        event_process.start()
        flir_process.start()
        show_process.start()

        # Wait for processes to finish
        try:

            event_process.join()
            flir_process.join()
            show_process.join()

        except KeyboardInterrupt:
            stop_event.set()
            event_process.terminate()
            flir_process.terminate()
            show_process.terminate()



