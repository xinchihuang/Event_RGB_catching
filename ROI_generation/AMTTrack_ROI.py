import cv2
import os
import sys
from multiprocessing import shared_memory
import numpy as np
import time
import torch

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config

# 注释掉自动配置检测，避免干扰不同的运行模式
# def ensure_correct_config():
#     """确保加载了正确的配置文件"""
#     current_bbox = Config.get_roi()['default_bbox']
#     # 如果是默认配置的bbox，尝试切换到config_l.yaml
#     if current_bbox == [430, 1180, 40, 40]:
#         try:
#             switch_global_config('config/config_l.yaml')
#             print("Auto-switched to config_l.yaml for development")
#         except:
#             pass

# # 自动配置检测 - 注释掉以避免干扰
# ensure_correct_config()
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
def model(shared_event_data, shared_flir_data,shared_ROI_data,num_event,system_state, stop_event):
    import sys
    import os
    # Add AMTTrack directory to path (it's in the parent directory)
    amttrack_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AMTTrack')
    sys.path.append(amttrack_path)
    from lib.test.evaluation.tracker import Tracker
    from spconv.pytorch.utils import PointToVoxel

    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    # 确保在子进程中加载正确的配置文件
    # 检查环境变量来确定应该使用哪个配置
    active_config = os.environ.get('ACTIVE_CONFIG_FILE')
    active_task = os.environ.get('ACTIVE_TASK')
    
    if active_config:
        print(f"Model process: Using config from environment: {active_config} (task: {active_task})")
        # Config will be automatically loaded based on environment variables
    else:
        print("Model process: No active config in environment, using default config")

    print(f"Model process using templates: {Config.get_paths()['template_npy']}, {Config.get_paths()['template_jpg']}")
    print(f"Model process using default_bbox: {Config.get_roi()['default_bbox']}")
    print(f"Model process using record_dir: {Config.get_paths()['record_dir']}")

    # Get image dimensions from config
    dims = Config.get_image_dimensions()
    rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']
    event_width, event_height = dims['event_width'], dims['event_height']
    
    # Get image processing parameters from config
    processing = Config.get_image_processing()
    
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=processing['fps'],
                                                       palette=ColorPalette.Dark)
    def detect_initial_bbox(image, event_data):

        """
        自动检测球的初始位置
        """
        # 导入球检测器
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        sys.path.append(current_dir)
        from initial_bbox.ball_detector import BallDetector

        # 初始化球检测器
        # ball_detector = BallDetector(min_radius=5, max_radius=40, param2=25)
        
        # 生成事件图像用于辅助检测和叠加显示
        event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame,
                                       accumulation_time_us=processing['accumulation_time_us'])
        event_image_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        
        # 将事件图像转换为二值图像以便更好的可视化
        _, binary = cv2.threshold(event_image_gray, processing['binary_threshold'], processing['binary_max_value'], cv2.THRESH_BINARY)
        event_image_viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 尝试自动检测球
        # detected_bbox = ball_detector.detect_ball(image, event_image_gray)
        # print(detected_bbox)
        detected_bbox = Config.get_roi()['default_bbox']
        print(detected_bbox)
        if detected_bbox is not None:
            x, y, w, h = detected_bbox
            print(f"自动检测到球的位置: x={x}, y={y}, w={w}, h={h}")

            # 可视化检测结果 - 在RGB图像上绘制bbox
            vis_image = image.copy()
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, "Auto Detected Ball", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 在事件图像上也绘制bbox
            cv2.rectangle(event_image_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(event_image_viz, "Event View", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 叠加显示RGB图像和事件图像
            combined = np.hstack((vis_image, event_image_viz))
            
            # 显示检测结果，让用户确认
            scale = processing['display_scale']
            height, width = combined.shape[:2]
            new_size = (int(width * scale), int(height * scale))
            resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
            cv2.imshow('Ball Detection (RGB + Event) - Press any key to continue', resized_img)
            cv2.waitKey(2000)  # 显示2秒
            cv2.destroyAllWindows()

            return x, y, w, h
        else:
            # 如果自动检测失败，使用默认位置
            print("自动检测失败，使用默认位置")
            roi_config = Config.get_roi()
            if 'backup_bbox' in roi_config:
                x, y, w, h = roi_config['backup_bbox']
            else:
                # 如果没有backup_bbox，使用default_bbox
                x, y, w, h = roi_config['default_bbox']
            return x, y, w, h

    device = torch.device("cuda")
    voxel_generator = PointToVoxel(
        # 给定每个voxel的长宽高  [0.05, 0.05, 0.1]
        vsize_xyz=[1000, 10, 10],  # [0.2, 0.25, 0.16]  # [50, 10, 10]  [50, 35, 26] 因此坐标范围（20,20,20）  (20, 34/35, 26)
        # 给定点云的范围 [  0.  -40.   -3.   70.4  40.    1. ]
        coors_range_xyz=[  0.,  0,   0,   100000,  event_width ,   event_height ],
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

    # 使用绝对路径确保在多进程环境中能找到文件 (77.npy在项目根目录)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_path = os.path.join(project_root, Config.get_paths()['template_npy'])
    first_event_data=np.load(npy_path)

    coor, features = transform_points_to_voxels(event_data=first_event_data, voxel_generator=voxel_generator,
                                                device=device)
    event_template = torch.cat((coor, features), dim=1)

    # event_template = torch.zeros([1024, 19])


    # 使用绝对路径确保在多进程环境中能找到文件 (77.jpg在项目根目录)
    jpg_path = os.path.join(project_root, Config.get_paths()['template_jpg'])
    first_flir_image=cv2.imread(jpg_path)
    
    # Apply brightness enhancement from config
    processing = Config.get_image_processing()
    brightness_multiplier = processing['brightness_multiplier']
    if brightness_multiplier != 1.0:
        first_flir_image = np.clip(first_flir_image.astype(np.float32) * brightness_multiplier, 0, 255).astype(np.uint8)

    initial_bbox = detect_initial_bbox(first_flir_image,first_event_data)
    # initial_bbox= (495, 1098, 40, 40)

    x, y, w, h= initial_bbox
    # cv2.rectangle(first_flir_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("a", first_flir_image)
    # cv2.waitKey(0)
    bbox = [x, y, w, h]
    init_info = {'init_bbox': bbox}

    count=0
    roi_params = Config.get_roi()
    roi_range = roi_params['roi_range']
    initialized= False
    system_state.value = 'w'  # 设置初始状态为等待
    while not stop_event.is_set():
        event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
        flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()
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
            event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf).copy()
            flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf).copy()

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
            bbox = out.get('target_bbox')
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

            bbox=out.get('target_bbox')
            confidence = out.get('confidence', 0.0)  # Get confidence, default to 0.0 if not available

            # Confidence-based tracking strategy
            if confidence < 0.15:  # Very low confidence - reset to image center
                center_x = rgb_width // 2
                center_y = rgb_height // 2
                center_size = 40
                bbox = [center_x - center_size//2, center_y - center_size//2, center_size, center_size]
                print(f"Frame {count}: Very low confidence {confidence:.3f}, reset to center: {bbox}")
                # Update info for next iteration
                info = {'previous_output': {'target_bbox': bbox}, 'gt_bbox': bbox}
            elif confidence < 0.2:  # Low confidence - keep current bbox but warn
                print(f"Frame {count}: Low confidence {confidence:.3f}, bbox: {bbox}")
            elif count % 30 == 0:  # Normal confidence - periodic logging
                print(f"Frame {count}: Normal confidence {confidence:.3f}, bbox: {bbox}")

            state = [int(s) for s in bbox]
            x_r, y_r, w_r, h_r = state

            ROI[0]=x_r
            ROI[1]=y_r
            ROI[2]=w_r
            ROI[3]=h_r
            event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
            event_frame = np.ascontiguousarray(event_frame)
            event_frame_gen.generate_frame(event_data, event_frame, 
                                       accumulation_time_us=processing['accumulation_time_us'])
            image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(image, processing['binary_threshold'], processing['binary_max_value'], cv2.THRESH_BINARY)
            event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            rgb_image = flir_data
            # All drawing code removed - no visual feedback needed
            
            combined = np.hstack((rgb_image, event_image))
            scale = processing['display_scale']
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