from colorsys import yiq_to_rgb

import cv2

import argparse
import os
import csv
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
def align_position(rgb_bbox,rgb_width=1080, rgb_height = 1440, event_width=720, event_height = 1280):
    x,y,w,h=rgb_bbox
    rgb_center_x=rgb_width/2
    rgb_center_y=rgb_height/2
    event_center_x=event_width/2
    event_center_y=event_height/2
    event_pixel=3.6
    rgb_pixel=4.86

    x_e = (x-rgb_center_x) * event_pixel / rgb_pixel + event_center_x-25
    y_e = (y-rgb_center_y) * event_pixel / rgb_pixel + event_center_y-30
    w_e = w * event_pixel/ rgb_pixel
    h_e = h * event_pixel / rgb_pixel
    return int(x_e), int(y_e), int(w_e), int(h_e)
def align_buffer():
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_core.event_io import EventsIterator
    event_iterator = iter(
        EventsIterator(input_path="/home/xinchi/Event_RGB_catching_data/saved_data/event/0.raw", mode="mixed",
                       delta_t=10000, n_events=10000))
    event_dtype = {
        'names': ['x', 'y', 'p', 't'],
        'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8],
        'itemsize': 16
    }
    event_shape = (2000,10000,)
    event_buffer = np.ndarray(event_shape, dtype=event_dtype)
    event_count = 0
    while True:
        start_time = time.perf_counter()
        if event_count>=2000:
            break
        try:
            event_data = next(event_iterator)
        except StopIteration:
            break
        event_data['x'], event_data['y'] = event_data['y'].copy(), -event_data['x'].copy()
        event_data['y'] += 1279
        # print(event_data['t'])
        dt = 1000
        event_buffer[event_count,:event_data.shape[0]] = event_data
        event_count += 1
    # print(event_buffer.shape)
    rgb_width, rgb_height = 1080, 1440
    event_width, event_height = 720, 1280
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_width, sensor_height=event_height,
                                                       fps=100,
                                                       palette=ColorPalette.Dark)

    video_path = "/home/xinchi/Event_RGB_catching_data/saved_data/rgb/0.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_count = 1
    print("align start")
    bbox=[]
    while rgb_count<total_frames:

        event_data = event_buffer[round(event_count*(rgb_count-3)/total_frames),:]
        # print(event_data.shape,int(event_count*rgb_count/total_frames))
        event_frame = np.zeros((event_height, event_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame)
        image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        event_image=cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # print("frame:", count)
        ret, frame = cap.read()
        cv_image = frame

        #
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 2. 定义目标颜色范围（以绿色为例）
        # 下面这个范围可以根据你的需求进行微调
        lower_green = np.array([ 25,  50, 100])  # 绿色下限
        upper_green = np.array([ 46, 200, 255])  # 绿色上限
        # 生成掩膜
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 3. 形态学操作（根据需要可调整或注释掉）
        # 去噪：先腐蚀再膨胀
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 4. 查找轮廓并绘制外接矩形
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # 可以根据面积大小进行过滤，避免过多小噪点
            if area > 300:  # 仅示例数值，可根据实际情况调整
                x, y, w, h = cv2.boundingRect(contour)
                x_e,y_e,w_e,h_e=align_position([x,y,w,h])
                # print(x,x_e)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 在原图上绘制红色矩形
                cv2.rectangle(event_image, (x_e, y_e), (x_e + w_e, y_e + h_e), (0, 0, 255), 2)  # 在原图上绘制红色矩形
                bbox.append([x, y, w, h])


        cv2.imshow('rgb ', event_image)
        cv2.waitKey(1)
        cv2.imwrite("/home/xinchi/Event_RGB_catching_data/dataset/rgb/" + str(rgb_count) + ".jpg", cv_image)
        cv2.imwrite("/home/xinchi/Event_RGB_catching_data/dataset/event/" + str(rgb_count) + ".jpg", event_image)
        rgb_count += 1
    with open("/home/xinchi/Event_RGB_catching_data/dataset/bbox/" + str(0) + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入每个 bounding box
        for box in bbox:
            writer.writerow(box)


align_buffer()