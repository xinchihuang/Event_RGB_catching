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

# from torch.fx.experimental import recording

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

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
            event_memory,event_dtype,event_shape=shared_event_data
            event_shm = shared_memory.SharedMemory(name=event_memory.name)
            # Attach to the shared memory and create a NumPy array view
            shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
            shm[:] = np.zeros(event_shape, dtype=event_dtype)
            shm[:event_data.shape[0]] = event_data
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

            flir_memory,flir_dtype,flir_shape=shared_flir_data
            flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
            shm = np.ndarray(cv_image.shape, dtype=flir_dtype, buffer=flir_shm.buf)
            shm[:] = cv_image
            image_result.Release()
        except Exception as e:
            print(f"Error in read_from_flir: {e}")
            continue
def read_from_raw(shared_event_data, stop_event):
    from metavision_core.event_io import EventsIterator

    event_iterator = iter(EventsIterator(input_path="/home/xinchi/Event_RGB_catching_data/saved_data/event/1.raw",mode="mixed", delta_t=1000, n_events=5000))
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

def record_event(shared_event_data, keyboard_input, stop_event):
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


    with MTWindow(title="Metavision Events Viewer", width=event_width, height=event_height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_height, sensor_height=event_width, fps=100,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            cd_frame = cv2.rotate(cd_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)


        while not stop_event.is_set():
            # event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

            key = keyboard_input.value
            # print(key)

            if key == 'r' and recording == False:
                recording = True
                print(f"Event recording toggled: {'ON' if recording else 'OFF'}")

                if device.get_i_events_stream():
                    out_path = os.path.join(save_dir, "event")
                    num_file = len(
                        [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))])
                    log_path = f"{out_path}/{num_file}.raw"

                    print(f'Recording to {log_path}')
                    device.get_i_events_stream().log_raw_data(log_path)
                else:
                    print("Event recording already initialized.")

            if key == 'f' and recording == True:
                recording = False
                print(f"Event recording toggled: {'ON' if recording else 'OFF'}")
                if device.get_i_events_stream():
                    device.get_i_events_stream().stop_log_raw_data()

            event_data = next(event_iterator)
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(event_data)
            # # Process event data
            # if event_data is not None:
            #     EventLoop.poll_and_dispatch()
            #     event_frame_gen.process_events(event_data)


def record_flir(shared_flir_data,keyboard_input, stop_event):
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
    # output_filename = 'output_video.mp4'
    rgb_width, rgb_height = 1080, 1440
    rgb_frame_rate = 100
    rgb_frame_size = (rgb_width, rgb_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Define codec and create VideoWriter object
    out_video = None  # Initialize as None to handle lazy initialization
    recording = False  # Flag to manage recording state
    save_dir = "/home/xinchi/Event_RGB_catching_data/saved_data"

    while not stop_event.is_set():
        image_result = cam.GetNextImage(10000)
        if image_result.IsIncomplete():
            continue
        image_data = image_result.GetNDArray()
        cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        flir_data = cv_image
        key = keyboard_input.value
        if key == 'r' and recording==False:
            recording = True
            print(f"Recording toggled: {'ON' if recording else 'OFF'}")

            if out_video is None:
                print("Initializing video recording...")
                out_path = os.path.join(save_dir, "rgb")
                num_file = len(
                    [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))])

                output_filename = os.path.join(out_path, f"{num_file}.mp4")
                out_video = cv2.VideoWriter(output_filename, fourcc, rgb_frame_rate, rgb_frame_size)
            else:
                print("Recording already initialized.")

        if key == 'f' and recording==True:
            recording = False
            print(f"Recording toggled: {'ON' if recording else 'OFF'}")
            if out_video:
                print("Stopping video recording...")
                out_video.release()
                out_video = None
        # Process FLIR data
        if flir_data is not None:
            # Resize flir_data if necessary
            if (flir_data.shape[1], flir_data.shape[0]) != rgb_frame_size:
                flir_data = cv2.resize(flir_data, rgb_frame_size)
            # Display the camera feed
            cv2.imshow('original_image', flir_data)
            # Write frame to video if recording
            if recording and out_video:
                out_video.write(flir_data)
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
    def on_press(key):
        try:
            # Update the shared value with the last key pressed
            print(f"Key {key} pressed")
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


    record=True
    if record==True:
        record_flir_process = Process(target=record_flir, args=(shared_flir_data,keyboard_input, stop_event))
        record_event_process = Process(target=record_event, args=(shared_event_data,keyboard_input, stop_event))
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



