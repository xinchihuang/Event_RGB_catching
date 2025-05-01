
import cv2
import csv
import sys
import numpy as np
import time
import os
import shutil


# from ekf_predict import ekf_predict
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def align_position(rgb_bbox,rgb_width=1080, rgb_height = 1440, event_width=720, event_height = 1280):
    x,y,w,h=rgb_bbox
    rgb_center_x=rgb_width/2
    rgb_center_y=rgb_height/2
    event_center_x=event_width/2
    event_center_y=event_height/2
    event_pixel=3.7
    rgb_pixel=4.86

    x_e = (x-rgb_center_x) * event_pixel / rgb_pixel + event_center_x-25
    y_e = (y-rgb_center_y) * event_pixel / rgb_pixel + event_center_y-30
    w_e = w * event_pixel/ rgb_pixel
    h_e = h * event_pixel / rgb_pixel
    return int(x_e), int(y_e), int(w_e), int(h_e)
def align_to_rgb(event_data,rgb_width=1080, rgb_height = 1440, event_width=720, event_height = 1280):
    x_e,y_e=event_data['x'].copy(),event_data['y'].copy()
    rgb_center_x=rgb_width/2
    rgb_center_y=rgb_height/2
    # print(rgb_center_x,rgb_center_y)
    event_center_x=event_width/2
    event_center_y=event_height/2
    event_pixel=3.1
    rgb_pixel=4.86

    x = (x_e-event_center_x) * rgb_pixel / event_pixel + rgb_center_x+40
    y = (y_e-event_center_y) * rgb_pixel / event_pixel + rgb_center_y-25
    x= np.clip(x, 0, rgb_width)
    y= np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)
def align_buffer(root_dir,index,bias=0):
    # gc.collect()
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_core.event_io import EventsIterator
    event_iterator = iter(
        EventsIterator(input_path=f"/home/xinchi/Event_RGB_catching_data/saved_data/event/{index}.raw", mode="mixed",
                       delta_t=10000, n_events=10000))
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape = (3000,10000,)
    event_buffer = np.ndarray(event_shape, dtype=event_dtype)
    event_count = 0
    while True:
        start_time = time.perf_counter()
        if event_count>=3000:
            break
        try:
            event_data = next(event_iterator)
        except StopIteration:
            break
        event_data['x'], event_data['y'] = event_data['y'].copy(), -event_data['x'].copy()
        event_data['y'] += 1279
        event_data['x'], event_data['y'] = align_to_rgb(event_data)
        # print(event_data['t'])
        dt = 500
        event_buffer[event_count,:event_data.shape[0]] = event_data
        event_count += 1
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=100,
                                                       palette=ColorPalette.Dark)

    video_path = f"/home/xinchi/Event_RGB_catching_data/saved_data/rgb/{index}.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = 30
    frame_time = 1.0 / fps
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # for _ in range(bias):
    #     _, _ = cap.read()
    rgb_frame_count = 0
    rgb_index=0
    event_index=0
    print("total rgb frames:", total_frames)
    print("total event frames:", event_count)
    print("align start")
    bbox=[]
    event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    if not os.path.exists(f"{root_dir}/{index}"):
        os.makedirs(f"{root_dir}/{index}")
        os.makedirs(f"{root_dir}/{index}/rgb")
        os.makedirs(f"{root_dir}/{index}/event")
    absent_list=[]
    if bias > 0:
        for _ in range(bias):
            _, _ = cap.read()
            rgb_frame_count += 1
    elif bias < 0:
        event_index += -bias
    while rgb_frame_count<total_frames:
        print("rgb frame:", rgb_index, "event frame:", event_index)
        start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        rgb_frame_count += 1

        cv_image = frame
        has_ball = False

        # print(cv_image.shape)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([ 25,  55, 100])  # 绿色下限
        upper_green = np.array([ 46, 200, 255])  # 绿色上限
        # 生成掩膜
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 300:  # 仅示例数值，可根据实际情况调整
                x, y, w, h = cv2.boundingRect(contour)
                # x_e,y_e,w_e,h_e=align_position([x,y,w,h])
                # print(x,x_e)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 在原图上绘制红色矩形
                bbox.append([x, y, w, h])
                has_ball=True
        print("processing event frame:", event_index)
        event_data = event_buffer[event_index * 25, :]
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame)
        image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if has_ball:
            absent_list.append(1)
            cv2.rectangle(event_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 在原图上绘制红色矩形
        else:
            absent_list.append(0)
        #
        # cv2.imshow('rgb ', frame)
        # cv2.waitKey(1)
        cv2.imwrite(f"{root_dir}/{index}/rgb/" + str(rgb_index) + ".jpg", cv_image)
        cv2.imwrite(f"{root_dir}/{index}/event/" + str(rgb_index) + ".jpg", event_image)
        rgb_index += 1
        event_index+=1
        elapsed_time = time.perf_counter() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
    # del event_buffer
    # gc.collect()
    shutil.copy(f"/home/xinchi/Event_RGB_catching_data/saved_data/event/{index}.raw", f"{root_dir}/{index}/event.raw")
    with open(f"{root_dir}/{index}/absent.txt", 'w', newline='') as f:
        for a in absent_list:
            f.write(str(a)+"\n")
    with open(f"{root_dir}/{index}/groundtruth.txt", 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入每个 bounding box
        for box in bbox:
            writer.writerow(box)

root_dir = "/home/xinchi/Event_RGB_catching_data/dataset"


# for index in range(12, 50):
index=12
align_buffer(root_dir,index,bias=-3)