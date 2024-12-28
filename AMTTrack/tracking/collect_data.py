import cv2
import os
import sys
import torch
import multiprocessing
import signal
import queue  # For Empty exception
import numpy as np
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def read_from_event(event_queue, stop_event):
    from metavision_core.event_io import EventsIterator

    # Initialize EventsIterator for live camera
    event_iterator = iter(EventsIterator(input_path="", delta_t=1000, n_events=100))
    while not stop_event.is_set():
        try:
            if event_queue.full():
                event_queue.get()  # Remove the oldest item
            event_data = next(event_iterator)
            event_queue.put(event_data, timeout=0.1)
        except StopIteration:
            # End of event stream
            stop_event.set()
            break
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"Error in read_from_event: {e}")
            continue

def read_from_flir(flir_queue, stop_event):
    import PySpin
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("No FLIR cameras detected.")
        system.ReleaseInstance()
        stop_event.set()
        return

    cam = cam_list[0]
    cam.Init()
    cam.BeginAcquisition()
    while not stop_event.is_set():
        try:
            image_result = cam.GetNextImage(1000)
            if image_result.IsIncomplete():
                continue
            image_data = image_result.GetNDArray()
            cv_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
            cv_image=cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            if flir_queue.full():
                flir_queue.get()
            flir_queue.put(cv_image, timeout=0.1)
            image_result.Release()
        except PySpin.SpinnakerException as e:
            print(f"PySpin error: {e}")
            continue
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"Error in read_from_flir: {e}")
            continue
    # Cleanup
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

def save_flir(event_queue, flir_queue, stop_event):
    import cv2
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import BaseWindow, MTWindow, UIKeyEvent, EventLoop

    width, height = 720, 1280
    output_filename = 'output_video.mp4'
    frame_rate = 30  # frames per second
    frame_size = (width, height)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out_video = None  # Initialize as None to handle lazy initialization

    recording = False  # Flag to manage recording state

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal recording, out_video
            if action != 1:  # Only handle key press, ignore release
                return
            try:
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    print("Exit key pressed.")
                    window.set_close_flag()
                    stop_event.set()
                elif key == ord('r') or key == UIKeyEvent.KEY_R:
                    recording = not recording
                    print(f"Recording toggled: {'ON' if recording else 'OFF'}")
                    if recording:
                        print("Initializing video recording...")
                        if out_video is None:
                            out_video = cv2.VideoWriter(output_filename, fourcc, frame_rate, frame_size)
                        else:
                            print("Recording already initialized.")
                    else:
                        print("Stopping video recording...")
                        if out_video:
                            out_video.release()
                            out_video = None
            except Exception as e:
                print(f"Error in keyboard callback: {e}")

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=height, sensor_height=width, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            cd_frame=cv2.rotate(cd_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        while not stop_event.is_set():
            try:
                event_data = event_queue.get(timeout=0.1)

            except queue.Empty:
                event_data = None
            try:
                flir_data = flir_queue.get(timeout=0.1)
            except queue.Empty:
                flir_data = None

            # Process event data
            if event_data is not None:
                EventLoop.poll_and_dispatch()
                event_frame_gen.process_events(event_data)

            # Process FLIR data
            if flir_data is not None:
                # Resize flir_data if necessary
                if (flir_data.shape[1], flir_data.shape[0]) != frame_size:
                    flir_data = cv2.resize(flir_data, frame_size)
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
def save_event(event_queue, flir_queue, stop_event):
    import numpy as np
    saved_events = []
    while not stop_event.is_set():
        try:
            event_data = event_queue.get(timeout=0.1)

        except queue.Empty:
            event_data = None
        # Process event data
        if event_data is not None:
            saved_events.append(event_data)

    saved_events_array=np.array(saved_events)
    np.save('saved_events.npy', saved_events_array)
def signal_handler(sig, frame):
    print("Terminating processes...")
    stop_event.set()
    sys.exit(0)

if __name__ == '__main__':
    device = torch.device("cuda")

    multiprocessing.set_start_method('spawn')

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C gracefully
    event_queue = multiprocessing.Queue(maxsize=5)
    flir_queue = multiprocessing.Queue(maxsize=5)
    stop_event = multiprocessing.Event()

    # Processes
    save_event_process = multiprocessing.Process(target=save_event, args=(event_queue, flir_queue, stop_event))
    save_flir_process = multiprocessing.Process(target=save_flir, args=(event_queue, flir_queue, stop_event))
    event_process = multiprocessing.Process(target=read_from_event, args=(event_queue, stop_event))
    flir_process = multiprocessing.Process(target=read_from_flir, args=(flir_queue, stop_event))
    save_flir_process.start()
    event_process.start()
    flir_process.start()

    # Wait for processes to finish
    try:
        save_flir_process.join()
        save_event_process.join()
    except KeyboardInterrupt:
        stop_event.set()
        save_event_process.terminate()
        save_flir_process.terminate()
        event_process.terminate()
        flir_process.terminate()
    finally:
        event_process.join()
        flir_process.join()
