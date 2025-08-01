import cv2
import os
import torch
from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import math
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
    if time_length == 0:
        time_length = 1e-6
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
def read_from_video(shared_flir_data, system_state, raw_count,
                    record_index, start_ts, stop_event,video_broadcasting,
                    record_dir='/home/xinchi/Event_RGB_catching_data/saved_data_l'):

    import os, time, math, cv2
    from multiprocessing import shared_memory
    import numpy as np

    count = 0
    initialized = False
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # ---------- 准备阶段：只在 state=='w' 且未初始化时执行 ----------
        if system_state.value == 'w' and not initialized:
            video_path = os.path.join(record_dir, f"rgb/{record_index.value}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 等待 raw 线程把 raw_count 写好，否则 ratio 计算错误  ### FIX-1
            while raw_count.value < total_frames and not stop_event.is_set():
                time.sleep(0.001)

            ratio = (raw_count.value - 1) / (total_frames - 1)

            period_raw = 0.002
            period_video = ratio * period_raw
            print(f"Frame Interval: {period_video:.4f} seconds")
            print(f"Total Frames: {total_frames}, Raw Count: {raw_count.value}, Ratio: {ratio:.4f}")

            # 预读到内存
            rgb_buffer = []
            ret, frame = cap.read()
            while ret:
                rgb_buffer.append(frame)
                ret, frame = cap.read()
            print("Video ready")

            count = 0
            initialized = True

        # ---------- 播放阶段 -----------------------------------------
        if system_state.value == 't' and initialized:
            if start_ts.value == 0.0:                       # 第一次写入壁钟
                start_ts.value = time.perf_counter()

            loop_idx = count
            target_time = start_ts.value + loop_idx * period_video

            if count >= len(rgb_buffer):                   # 播完
                video_broadcasting.value = 'f'
                initialized = False                        # 只在真正结束时清零 ### FIX-2
                print("End of video data.")
                continue

            cv_image = rgb_buffer[count]

            # 写共享内存
            shm = np.ndarray(cv_image.shape, dtype=flir_dtype, buffer=flir_shm.buf)
            shm[:] = cv_image

            # 睡到周期边界
            sleep_time = target_time + period_video - time.perf_counter()  ### FIX-3
            if sleep_time > 0:
                time.sleep(sleep_time)

            count += 1

def pure_RGB(shared_event_data, shared_flir_data,shared_ROI_data,num_event,system_state, stop_event):

    rgb_width, rgb_height = 1080, 1440
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
    # time.sleep(2)
    # system_state.value = 't'

    count = 1

    while not stop_event.is_set():
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
        if num_event.value == 0:
            continue
        if system_state.value == 't':
            # print(count)
            start_time = time.time()
            # Convert from BGR to HSV color space

            rgb_image=flir_data
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            lower_hsv = np.array([30, 70, 100])  # H,S,V
            upper_hsv = np.array([50, 255, 255])

            # 直接在 BGR 上 inRange
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Proceed only if at least one contour was found

            if len(contours) > 0:
                # Find the largest contour by area
                c = max(contours, key=cv2.contourArea)
                # Get bounding box coordinates from the largest contour
                x, y, w, h = cv2.boundingRect(c)
                state=[x,y,w,h]
                x_r, y_r, w_r, h_r = state
                ROI[0] = x_r
                ROI[1] = y_r
                ROI[2] = w_r
                ROI[3] = h_r
                rgb_image = flir_data
                # cv2.rectangle(event_image, (x_r, y_r), (x_r + h_r, y_r + w_r), (0, 255, 0), 2)
                cv2.rectangle(rgb_image, (x_r, y_r), (x_r + w_r, y_r+ h_r), (0, 255, 0), 2)

            combined = rgb_image
            scale = 0.8
            # 获取新的尺寸
            width = int(combined.shape[1] * scale)
            height = int(combined.shape[0] * scale)
            new_size = (width, height)

            resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
            # cv2.imshow('Image with Bounding Box', resized_img)
            # cv2.waitKey(1)
            # cv2.imwrite(f"{root_dir}/{0}/combined/" + str(count) + ".jpg", combined)
            end_t = time.time()
            count += 1
def RGB_tracking_3D(shared_event_data,shared_flir_data,shared_ROI_data,shared_ball_pixel_3D_data,ball_observations,num_events,system_state, stop_event):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    rgb_width, rgb_height = 1080, 1440

    ball_pixel_3D_memory,ball_pixel_3D_dtype,ball_pixel_3D_shape = shared_ball_pixel_3D_data
    ball_pixel_3D_shm = shared_memory.SharedMemory(name=ball_pixel_3D_memory.name)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype, buffer=ball_pixel_3D_shm.buf)
    ROI_data_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_data_memory.name)
    count=0

    while system_state.value == 'w':
        continue
    print("start event tracking")
    p_x, p_y, p_w, p_h = 0,0,0,0
    while not stop_event.is_set():
        if system_state.value == 'e':
            break

        # print(system_state.value)
        if system_state.value == 't':
            start_t=time.time()
            try:
                raw_event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

                flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
                # print(num_events.value)
                if num_events.value == 0:
                    continue
                event_data = raw_event_data[:num_events.value]
                t = event_data['t'][-1] / 1000000
                ROI_data = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)
                x,y,w,h= ROI_data
                rgb_image = flir_data
                if not p_x==x or not p_y==y or not p_w==w or not p_h==h:

                    cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
                    # cv2.rectangle(event_image, (535,715), (545, 725), (0, 255, 0), 2)
                    # cv2.rectangle(event_image, (x_r, y_r), (x_r + h_r, y_r + w_r), (255, 255, 0), 2)
                    ball_num = ball_observations.value
                    ball_pixel_3D[ball_num][0] = float(x + h / 2)
                    ball_pixel_3D[ball_num][1] = float(y + h / 2)
                    ball_pixel_3D[ball_num][2] = float(h)
                    ball_pixel_3D[ball_num][3] = t
                    if ball_num < 100:
                        ball_num += 1
                    print(ball_num)
                    ball_observations.value = ball_num
                    p_x, p_y, p_w, p_h = x, y, w, h

                combined = rgb_image
                scale = 0.8

                # 获取新的尺寸
                width = int(combined.shape[1] * scale)
                height = int(combined.shape[0] * scale)
                new_size = (width, height)

                # 缩放图像（使用 INTER_AREA 插值更适合缩小）
                resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('Image with Bounding Box', resized_img)
                cv2.waitKey(1)
                # cv2.imwrite(f"{save_dir}/{0}/event/" + str(count) + ".jpg", combined)
                # print("iters:",count)
                # print("ball_observations",ball_observations.value)

                count += 1
                end_t = time.time()
                # print("Event time:", end_t - start_t)
            except Exception as e:
                # print(f"Error in event tracking: {e}")
                continue
def model(shared_event_data, shared_flir_data,shared_ROI_data,num_event,system_state, stop_event):
    from lib.test.evaluation.tracker import Tracker
    from spconv.pytorch.utils import PointToVoxel

    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=0.0,
                                                       palette=ColorPalette.Dark)
    def draw_initial_bbox(image,event_data):
        x,y,w,h = 470,1145,40,40
        # rgb_image = image
        # event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        # event_frame = np.ascontiguousarray(event_frame)
        # event_frame_gen.generate_frame(event_data, event_frame, accumulation_time_us=1000)
        # event_image_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(event_image_gray, 40, 255, cv2.THRESH_BINARY)
        # event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        #
        # cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
        # cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
        # combined = np.hstack((rgb_image, event_image))
        # scale = 0.8
        # # 获取新的尺寸
        # width = int(combined.shape[1] * scale)
        # height = int(combined.shape[0] * scale)
        # new_size = (width, height)
        #
        # resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
        #
        # cv2.imshow('Image with Bounding Box', resized_img)
        # cv2.waitKey(0)
        return x, y, w, h

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

    start_time = time.time()


    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)

    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    first_flir_image = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

    ROI_memory, ROI_dtype, ROI_shape = shared_ROI_data
    ROI_shm = shared_memory.SharedMemory(name=ROI_memory.name)
    ROI = np.ndarray(ROI_shape, dtype=ROI_dtype, buffer=ROI_shm.buf)

    first_event_data=np.load("/home/xinchi/Event_RGB_catching/AMTTrack/77.npy")

    coor, features = transform_points_to_voxels(event_data=first_event_data, voxel_generator=voxel_generator,
                                                device=device)
    event_template = torch.cat((coor, features), dim=1)

    # event_template = torch.zeros([1024, 19])


    first_flir_image=cv2.imread("/home/xinchi/Event_RGB_catching/AMTTrack/77.jpg")
    initial_bbox = draw_initial_bbox(first_flir_image,first_event_data)
    x, y, w, h = initial_bbox
    bbox = [x, y, w, h]
    init_info = {'init_bbox': bbox}

    count=0
    roi_range=500
    initialized= False
    while not stop_event.is_set():
        event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
        event_data = event_data[:num_event.value]

        if system_state.value == 'w' and not initialized:
            print('Model Initializing')
            # AMtracker = Tracker('amttrack', 'amttrack_felt', 'felt', None)
            AMtracker = Tracker('amttrack', 'amttrack_ball', 'ball', None)

            params = AMtracker.get_parameters()
            # print(params)
            params.debug = None
            tracker = AMtracker.create_tracker(params)

            out = tracker.initialize(first_flir_image, event_template, init_info, idx=0)
            # run one iter
            event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
            flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)

            start_time = time.time()
            x, y, w, h = initial_bbox
            bbox = [x, y, w, h]
            init_info = {'init_bbox': bbox}
            info = {'previous_output': {'target_bbox': bbox}, 'gt_bbox': bbox}
            voxel_start = time.time()
            coor, features = transform_points_to_voxels(event_data=event_data, voxel_generator=voxel_generator,
                                                        device=device)
            voxel_end = time.time()
            print("Voxel time:", voxel_end - voxel_start)
            # print(coor.shape, features.shape)
            event_search = torch.cat((coor, features), dim=1)
            tracker.state = bbox
            inference_start = time.time()
            out = tracker.track(flir_data, event_search, info)
            inference_end = time.time()
            print("Inference time:", inference_end - inference_start)
            # print("OUT",out)
            bbox = out['target_bbox']
            print("Model Ready")
            print("")
            initialized = True
            count=0
            system_state.value = 't'

        if num_event.value == 0:
            continue
        if system_state.value == 't':
            # print(count)
            start_time = time.time()
            info = {'previous_output':{'target_bbox':bbox},'gt_bbox':bbox}
            # info['previous_output'] = None
            voxel_start=time.time()
            try:

                coor, features = transform_points_to_voxels(event_data=event_data, voxel_generator=voxel_generator,
                                                            device=device)
            except:
                continue
            voxel_end=time.time()
            # print("Voxel time:",voxel_end-voxel_start)
            # print(coor.shape, features.shape)
            event_search =  torch.cat((coor, features), dim=1)
            if event_search==None:
                continue
            tracker.state=bbox
            inference_start=time.time()
            out = tracker.track(flir_data, event_search, info)
            inference_end=time.time()

            bbox=out['target_bbox']

            state = [int(s) for s in bbox]
            x_r, y_r, w_r, h_r = state
            print(state)

            ROI[0]=x_r
            ROI[1]=y_r
            ROI[2]=w_r
            ROI[3]=h_r
            event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
            event_frame = np.ascontiguousarray(event_frame)
            event_frame_gen.generate_frame(event_data, event_frame, accumulation_time_us=1000)
            image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
            event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            rgb_image=flir_data
            # cv2.rectangle(event_image, (x_r, y_r), (x_r + h_r, y_r + w_r), (0, 255, 0), 2)
            cv2.rectangle(rgb_image, (x_r, y_r), (x_r + h_r, y_r+ w_r), (0, 255, 0), 2)
            combined = np.hstack((rgb_image, event_image))
            scale = 0.8
            # 获取新的尺寸
            width = int(combined.shape[1] * scale)
            height = int(combined.shape[0] * scale)
            new_size = (width, height)

            resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
            #
            # cv2.imshow('Image with Bounding Box', resized_img)
            # cv2.waitKey(1)
            # cv2.imwrite(f"{root_dir}/{0}/combined/" + str(count) + ".jpg", combined)
            end_t = time.time()
            count+=1
            initialized=False
