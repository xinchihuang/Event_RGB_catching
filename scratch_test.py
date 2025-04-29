from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from sklearn.cluster import DBSCAN
from metavision_core.event_io import EventsIterator
import numpy as np
import cv2
import os
def read_from_raw(index):
    event_iterator = iter(EventsIterator(input_path=f"/home/xinchi/Event_RGB_catching_data/saved_data/event/{index}.raw",mode="mixed", delta_t=10000, n_events=10000))
    event_width, event_height = 720, 1280
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=event_width, sensor_height=event_height,
                                                       fps=100,
                                                       palette=ColorPalette.Dark)
    ball_pixel_3D_dtype = np.float64
    ball_pixel_3D_shape = (1000, 4)
    ball_pixel_3D = np.ndarray(ball_pixel_3D_shape, dtype=ball_pixel_3D_dtype)
    ball_pixel_3D[:] = np.zeros(ball_pixel_3D_shape)
    event_count=0
    ball_num=0
    while True:
        # print(event_count)
        event_count+=1

        try:
            event_data = next(event_iterator)
        except StopIteration:
            break
        if not event_count % 10 == 0:
            continue
        event_data['x'],event_data['y']=event_data['y'].copy(),-event_data['x'].copy()
        event_data['y']+=1279
        t = event_data['t'][-1]/1000000
        # print(t)
        event_frame = np.zeros((event_height, event_width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        event_frame_gen.generate_frame(event_data, event_frame)
        image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        points = np.column_stack(np.where(binary > 0))  # 取所有非零像素点的坐标
        if points.shape[0] == 0:
            continue
        dbscan = DBSCAN(eps=12, min_samples=100).fit(points)
        labels = dbscan.labels_
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for cluster_id in set(labels):

            if cluster_id == -1:  # -1 代表噪声点，不处理
                continue
            # print(labels)
            # if len(labels)>1:
            #     continue
            print("ball:", ball_num, t)
            cluster_points = points[labels == cluster_id]
            # print(len(cluster_points))
            y, x, w, h = cv2.boundingRect(cluster_points)
            # y = min(cluster_points[:, 0])
            # x = min(cluster_points[:, 1])
            # w = max(cluster_points[:, 0]) - y
            # h = max(cluster_points[:, 1]) - x
            cv2.rectangle(color_image, (x, y), (x + h, y + w), (0, 255, 0), 2)

            ball_pixel_3D[ball_num][0] = float(x + h / 2)
            ball_pixel_3D[ball_num][1] = float(y + h / 2)
            ball_pixel_3D[ball_num][2] = float(h)
            ball_pixel_3D[ball_num][3] = t
            # print(time.time())
            if ball_num < 1000:
                ball_num += 1
                cv2.imwrite(f'/home/xinchi/Event_RGB_catching_data/frames/{ball_num}.jpg', color_image)
            break
        cv2.imshow('event ', color_image)
        cv2.waitKey(1)
        # print(count)
        event_count += 1
    save_dir = "/home/xinchi/Event_RGB_catching_data/ball_pixels_3D"
    num_file = len(
        [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
    save_path = f"{save_dir}/{num_file}.npy"
    filtered_arr = ball_pixel_3D[~np.any(ball_pixel_3D == 0, axis=1)]
    print(filtered_arr)
    print("save to:", save_path)
    np.save(save_path, filtered_arr)
if __name__ == '__main__':
    for i in range(50):
        read_from_raw(i)

