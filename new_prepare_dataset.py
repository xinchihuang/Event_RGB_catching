from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import multiprocessing
from AMTTrack.event_process import *
from AMTTrack.RGB_process import *
from AMTTrack.fitting_process import *
# def check_align(shared_event_data,shared_flir_data,num_event,system_state,stop_event):
#     from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
#     from sklearn.cluster import DBSCAN
#
#     event_memory, event_dtype, event_shape = shared_event_data
#     event_shm = shared_memory.SharedMemory(name=event_memory.name)
#     rgb_width, rgb_height = 1080, 1440
#     event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
#                                                        fps=0.0,
#                                                        palette=ColorPalette.Dark)
#
#     print(event_frame_gen.get_accumulation_time_us())
#     print(event_frame_gen.get_fps())
#     flir_memory, flir_dtype, flir_shape = shared_flir_data
#     flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
#     dbscan = DBSCAN(eps=9, min_samples=70)
#     count=0
#     # time.sleep(1)
#     root_dir = "/home/xinchi/Event_RGB_catching_data/dataset"
#     index=len([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
#     if not os.path.exists(f"{root_dir}/{index}"):
#         os.makedirs(f"{root_dir}/{index}")
#         os.makedirs(f"{root_dir}/{index}/combined")
#         os.makedirs(f"{root_dir}/{index}/rgb")
#         os.makedirs(f"{root_dir}/{index}/event")
#     time.sleep(5)
#     print("start tracking")
#     system_state.value='t'
#     buffered_frame=np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
#     while not stop_event.is_set():
#         if system_state.value == 'e':
#             break
#         # start_time = time.time()
#         flir_data = np.ndarray(flir_shape, dtype=flir_dtype, buffer=flir_shm.buf)
#         event_data = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
#         if num_event.value==0:
#             continue
#         event_data= event_data[:num_event.value]
#         rgb_image=flir_data
#         event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
#         event_frame = np.ascontiguousarray(event_frame)
#         event_frame_gen.generate_frame(event_data, event_frame,accumulation_time_us=1000)
#         image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
#         points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
#         dbscan.fit(points)
#         labels = dbscan.labels_
#         event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#         cluster_points = points[labels != -1]
#         if not len(cluster_points) == 0:
#             y = min(cluster_points[:, 0])
#             x = min(cluster_points[:, 1])
#             w = max(cluster_points[:, 0]) - y
#             h = max(cluster_points[:, 1]) - x
#             cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
#             cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
#         cv2.rectangle(rgb_image, (50, 50), (150, 150), (255, 255, 0), 2)
#         cv2.rectangle(rgb_image, (50, 950), (150, 1050), (255, 255, 0), 2)
#         cv2.rectangle(rgb_image, (930, 50), (1030, 150), (255, 255, 0), 2)
#         cv2.rectangle(rgb_image, (930, 950), (1030, 1050), (255, 255, 0), 2)
#         cv2.rectangle(rgb_image, (500, 1200), (600, 1300), (255, 255, 0), 2)
#         cv2.rectangle(event_image, (50, 50), (150, 150), (255, 255, 0), 2)
#         cv2.rectangle(event_image, (50, 950), (150, 1050), (255, 255, 0), 2)
#         cv2.rectangle(event_image, (930, 50), (1030, 150), (255, 255, 0), 2)
#         cv2.rectangle(event_image, (930, 950), (1030, 1050), (255, 255, 0), 2)
#         cv2.rectangle(event_image, (500, 1200), (600, 1300), (255, 255, 0), 2)
#         combined = np.hstack((rgb_image, event_image))
#         scale = 0.8
#         # 获取新的尺寸
#         width = int(combined.shape[1] * scale)
#         height = int(combined.shape[0] * scale)
#         new_size = (width, height)
#
#         resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
#
#         cv2.imshow('Image with Bounding Box', resized_img)
#         cv2.waitKey(1)
#         cv2.imwrite(f"{root_dir}/{0}/combined/" + str(count) + ".jpg", combined)
#         # cv2.imwrite(f"{root_dir}/{index}/event/" + str(rgb_index) + ".jpg", event_image)
#
#         count += 1
def save_data(shared_event_data,shared_flir_data,num_event,system_state,stop_event,index=0):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from sklearn.cluster import DBSCAN
    import shutil

    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_width, sensor_height=rgb_height,
                                                       fps=0.0,
                                                       palette=ColorPalette.Dark)
    flir_memory, flir_dtype, flir_shape = shared_flir_data
    flir_shm = shared_memory.SharedMemory(name=flir_memory.name)
    dbscan = DBSCAN(eps=9, min_samples=70)
    count=0
    # time.sleep(1)
    root_dir = "/home/xinchi/Event_RGB_catching_data/dataset"
    # index=len([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    if not os.path.exists(f"{root_dir}/{index}"):
        os.makedirs(f"{root_dir}/{index}")
        os.makedirs(f"{root_dir}/{index}/combined")
        os.makedirs(f"{root_dir}/{index}/rgb")
        os.makedirs(f"{root_dir}/{index}/event")
    absent_list = []
    time.sleep(5)
    print("start tracking")
    system_state.value='t'
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

            cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
            cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
        combined = np.hstack((rgb_image, event_image))
        scale = 0.8
        # 获取新的尺寸
        width = int(combined.shape[1] * scale)
        height = int(combined.shape[0] * scale)
        new_size = (width, height)

        resized_img = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

        cv2.imshow('Image with Bounding Box', resized_img)
        cv2.waitKey(1)
        cv2.imwrite(f"{root_dir}/{index}/rgb/" + str(count) + ".jpg", rgb_image)
        # np.save(f"{root_dir}/{index}/event/" + str(count) + ".npy", event_data)
        cv2.imwrite(f"{root_dir}/{index}/event/" + str(count) + ".jpg", event_image)
        #
        count += 1
        shutil.copy(f"/home/xinchi/Event_RGB_catching_data/saved_data_lab/event/{index}.raw",
                    f"{root_dir}/{index}/event.raw")
        # with open(f"{root_dir}/{index}/absent.txt", 'w', newline='') as f:
        #     for a in absent_list:
        #         f.write(str(a) + "\n")
        # with open(f"{root_dir}/{index}/groundtruth.txt", 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     # 写入每个 bounding box
        #     for box in bbox:
        #         writer.writerow(box)

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
if __name__ == '__main__':
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
    keyboard_input = Value('u', 'e')
    system_state=Value('u', 'w')
    raw_count = Value('i', 0)
    num_event = Value('i', 0)
    record_index = Value('i', 0)
    start_ts = Value('d', 0.0)
    stop_event = multiprocessing.Event()
    event_process = Process(target=read_from_raw,
                            args=(shared_event_data, system_state, num_event, raw_count, record_index, start_ts,
                                  stop_event))
    flir_process = Process(target=read_from_video,
                           args=(shared_flir_data, system_state, raw_count, record_index, start_ts, stop_event))
    # checker_process=Process(target=check_align,args=(shared_event_data,shared_flir_data,num_event,system_state,stop_event))
    save_process = Process(target=save_data,
                              args=(shared_event_data, shared_flir_data, num_event, system_state, stop_event))
    keyboard_process = Process(target=keyboard, args=(keyboard_input, system_state))

    event_process.start()
    flir_process.start()
    # checker_process.start()
    save_process.start()