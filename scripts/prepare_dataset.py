import math

import cv2
import csv
import sys
import numpy as np
import time
import os
import shutil
import multiprocessing
import scipy
from spconv.pytorch.utils import PointToVoxel
from sklearn.cluster import DBSCAN
from AMTTrack.event_process import *
from AMTTrack.RGB_process import *
from AMTTrack.fitting_process import *

# from ekf_predict import ekf_predict
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

# def align_to_rgb(event_data,rgb_width=1080, rgb_height = 1440, event_width=720, event_height = 1280):
#     x_e,y_e=event_data['x'].copy(),event_data['y'].copy()
#     rgb_center_x=rgb_width/2
#     rgb_center_y=rgb_height/2
#     # print(rgb_center_x,rgb_center_y)
#     event_center_x=event_width/2
#     event_center_y=event_height/2
#     event_pixel=3.1
#     rgb_pixel=4.86
#
#     x = (x_e-event_center_x) * rgb_pixel / event_pixel + rgb_center_x+40
#     y = (y_e-event_center_y) * rgb_pixel / event_pixel + rgb_center_y-25
#     x= np.clip(x, 0, rgb_width)
#     y= np.clip(y, 0, rgb_height)
#     return x.astype(int), y.astype(int)
def align_buffer_old(root_dir,index,bias=0):
    # gc.collect()
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_core.event_io import EventsIterator
    event_iterator = iter(
        EventsIterator(input_path=f"/home/xinchi/Event_RGB_catching_data/saved_data/event/{index}.raw", mode="mixed",
                       delta_t=2000, n_events=10000))
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
    # if bias > 0:
    #     for _ in range(bias):
    #         _, _ = cap.read()
    #         rgb_frame_count += 1
    # elif bias < 0:
    #     event_index += -bias
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
        event_index += 1
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
def transform_points_to_voxels(data_dict, voxel_generator, device,
                               save_voxel=5_000):
    """略微精简，并使 save_voxel 显式可调"""
    pts = data_dict['points'].to(device)
    shuffle = torch.randperm(pts.shape[0], device=device)
    pts = pts[shuffle]

    voxels, coords, num_pts = voxel_generator(pts)
    # 选密度最高的前 save_voxel 个 voxel
    if num_pts.shape[0] > save_voxel:
        _, idx = torch.topk(num_pts, save_voxel)
        voxels, coords, num_pts = voxels[idx], coords[idx], num_pts[idx]

    coords = coords.to(device)
    coords[:, [0, 1, 2]] = coords[:, [2, 1, 0]]  # (y,x,t)→(t,x,y)
    feats = voxels[:, :, 3]                      # 取极性通道

    return coords, feats
def align_buffer(root_dir,raw_data_dir,index,bias=0,skip_start=0,skip_end=0):
    # gc.collect()
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_core.event_io import EventsIterator
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("AAA")
    event_iterator = iter(
        EventsIterator(input_path=f"{raw_data_dir}/event/{index}.raw", mode="delta_t",
                       delta_t=2000, n_events=1000000))
    voxel_gen = PointToVoxel(
        vsize_xyz=[50, 10, 10],
        coors_range_xyz=[0, 0, 0, 1000, 345, 259],
        num_point_features=4,
        max_num_voxels=16000,
        max_num_points_per_voxel=16,
        device=device)
    save_voxel = 5000

    event_buffer=[]
    for ev in event_iterator:
        ev['x'], ev['y'] = ev['y'].copy(), -ev['x'].copy()
        ev['y'] += 1279
        ev['x'], ev['y'] = align_to_rgb(ev)
        event_buffer.append(ev)
    mod = len(event_buffer)

    event_buffer = event_buffer[:] if mod != 0 else event_buffer
    event_count = len(event_buffer)
    video_path = f"{raw_data_dir}/rgb/{index}.mp4"

    cap = cv2.VideoCapture(video_path)
    rgb_buffer = []
    ret, frame = cap.read()
    while ret:
        rgb_buffer.append(frame)
        ret, frame = cap.read()

    rgb_count = len(rgb_buffer)
    print("total rgb frames:", rgb_count)
    print("total event frames:", event_count)
    print("align start")

    frame_index = 0
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=100,
                                                       palette=ColorPalette.Dark)
    dbscan = DBSCAN(eps=12, min_samples=60)
    if os.path.exists(f"{root_dir}/{index}"):
        shutil.rmtree(f"{root_dir}/{index}")
    if not os.path.exists(f"{root_dir}/{index}"):
        os.makedirs(f"{root_dir}/{index}")
        os.makedirs(f"{root_dir}/{index}/rgb")
        os.makedirs(f"{root_dir}/{index}/event")
        os.makedirs(f"{root_dir}/{index}/combined")
        os.makedirs(f"{root_dir}/{index}/voxel")

    absent_list=[]
    bbox = []
    ratio=(event_count-bias)/(rgb_count-1)
    if ratio > 10:
        ratio=10
    else:
        rgb_buffer= rgb_buffer[rgb_count-(event_count-bias)//10:]
        # skip_start=rgb_count-(event_count-bias)//10

    while True:
        if frame_index<=skip_start:
            frame_index += 1
            continue
        if frame_index>rgb_count-skip_end:
            print(rgb_count-skip_end)
            break

        if frame_index >= rgb_count or frame_index * ratio >= event_count or frame_index >= len(rgb_buffer):
            break
        rgb_image = rgb_buffer[frame_index]

        event_data= event_buffer[int(frame_index * ratio)]
        event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame)
        event_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(event_gray, 40, 255, cv2.THRESH_BINARY)
        points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
        if points.shape[0] > 0:
            np.set_printoptions(threshold=np.inf, linewidth=200)
            dbscan.fit(points)
            labels = dbscan.labels_
            cluster_points = points[labels != -1]
        else:
            cluster_points = np.array([]).reshape(0, 2)  # 如果没有点，创建一个空的数组

        event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([30, 90, 50])  # H,S,V
        upper_hsv = np.array([50, 255, 255])

        # 直接在 BGR 上 inRange
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_range=10
        absent= False
        if len(contours) > 0 and len(cluster_points) > 0:
            c = max(contours, key=cv2.contourArea)
            x_r, y_r, w_r, h_r = cv2.boundingRect(c)
            y_e = min(cluster_points[:, 0])
            x_e = min(cluster_points[:, 1])
            w_e = max(cluster_points[:, 0])- y_e
            h_e = max(cluster_points[:, 1])- x_e
            x=min(x_r, x_e)
            y=min(y_r, y_e)
            w=max(x_r + w_r, x_e + w_e) - x
            h=max(y_r + h_r, y_e + h_e) - y

            if w>100 or h>100 or y> 1280:
                x, y, w, h = 0, 0, 0, 0
                bbox.append([x, y, w, h])
                absent_list.append(0)
            else:
                bbox.append([x, y, w, h])
                absent_list.append(1)
        elif len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > 100 or h > 100 or y> 1280:
                x, y, w, h = 0, 0, 0, 0
                bbox.append([x, y, w, h])
                absent_list.append(0)
            else:
                bbox.append([x, y, w, h])
                absent_list.append(1)

        elif len(cluster_points)>0:
            y = min(cluster_points[:, 0])
            x = min(cluster_points[:, 1])
            w = max(cluster_points[:, 0])-y
            h = max(cluster_points[:, 1])-x
            if w > 100 or h > 100 or y> 1280:
                x, y, w, h = 0, 0, 0, 0
                bbox.append([x, y, w, h])
                absent_list.append(0)
            else:
                bbox.append([x, y, w, h])
                absent_list.append(1)
        else:
            absent = True
            x, y, w, h = 0, 0, 0, 0
            bbox.append([x, y, w, h])
            absent_list.append(0)

        x=x-bbox_range
        y=y-bbox_range
        w=w+2*bbox_range
        h=h+2*bbox_range
        cv2.rectangle(rgb_image, (x, y), (x+ w, y + h), (0, 255, 0), 2)
        cv2.rectangle(event_image, (x, y), (x+ w, y + h), (0, 255, 0), 2)




        alpha = 0.7  # 0 = 全透明（看不到前景），1 = 不透明

        # combined = cv2.addWeighted(event_image, alpha, rgb_image, 1 - alpha, 0)
        combined = np.hstack((rgb_image, event_image))
        cv2.imshow('combined', combined)
        cv2.waitKey(1)
        # combined = np.hstack((rgb_image, event_image))
        cv2.imwrite(f"{root_dir}/{index}/rgb/" + str(frame_index-skip_start) + ".jpg", rgb_image)
        cv2.imwrite(f"{root_dir}/{index}/event/" + str(frame_index-skip_start) + ".jpg", event_image)
        cv2.imwrite(f"{root_dir}/{index}/combined/" + str(frame_index-skip_start) + ".jpg", combined)

        if event_data['t'].shape[0] == 0:
            print("No events in this frame, skipping...")
            frame_index += 1
            continue
        t_event = torch.as_tensor(event_data["t"], dtype=torch.float32, device=device).unsqueeze(1)

        x_event = torch.as_tensor(event_data["x"].astype(np.int16), device=device).unsqueeze(1)
        y_event = torch.as_tensor(event_data["y"].astype(np.int16), device=device).unsqueeze(1)

        p_event = torch.as_tensor(event_data["p"].astype(np.int8), device=device).unsqueeze(1)
        p_event[p_event == 0] = -1  # 负极标为 -1
        dt = t_event[-1] - t_event[0]
        t_event = ((t_event - t_event[0]) / dt * 1000) if dt > 0 else torch.zeros_like(t_event)

        event_points = torch.cat((t_event, x_event, y_event, p_event), dim=1)  # (N,4)
        coor, feats = transform_points_to_voxels(
            {'points': event_points},
            voxel_generator=voxel_gen,
            device=device,
            save_voxel=save_voxel  # 传参而不是用全局
        )

        scipy.io.savemat(
            os.path.join(f'{root_dir}/{index}/voxel', f"{frame_index-skip_start}.mat"),
            {"coor": coor.cpu().numpy(),
             "features": feats.cpu().numpy()}
        )

        frame_index+=1


    print("align finished")
    shutil.copy(f"{raw_data_dir}/event/{index}.raw", f"{root_dir}/{index}/event.raw")
    with open(f"{root_dir}/{index}/absent.txt", 'w', newline='') as f:
        for a in absent_list:
            f.write(str(a)+"\n")
    with open(f"{root_dir}/{index}/groundtruth.txt", 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入每个 bounding box
        for box in bbox:
            writer.writerow(box)
    with open(f"{root_dir}/{index}/adjust.txt", 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入每个 bounding box
        writer.writerow([bias, skip_start, skip_end])


root_dir = "/media/xinchi/data/Event_RGB_catching_data/dataset"
raw_data_dir = "/media/xinchi/data/Event_RGB_catching_data/saved_data_d"


# for index in range(12, 50):
# index=1
for index in range(10,100):
    align_buffer(root_dir,raw_data_dir,index,bias=0,skip_start=0,skip_end=0)