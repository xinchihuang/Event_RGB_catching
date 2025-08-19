import os
import csv
import cv2
import time
import shutil
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from spconv.pytorch.utils import PointToVoxel
import scipy.io as sio

# ---------------------------------------------------------------------
# Optional: if you already have align_to_rgb elsewhere, you can remove this.
# This version matches your commented code (with tweakable offsets).
def align_to_rgb(event_data, rgb_width=1080, rgb_height=1440,
                 event_width=720, event_height=1280,
                 event_pixel=3.1, rgb_pixel=4.86,
                 dx=40, dy=-25):
    x_e = event_data['x'].astype(np.float32, copy=False)
    y_e = event_data['y'].astype(np.float32, copy=False)

    rgb_center_x = rgb_width / 2.0
    rgb_center_y = rgb_height / 2.0
    event_center_x = event_width / 2.0
    event_center_y = event_height / 2.0

    x = (x_e - event_center_x) * (rgb_pixel / event_pixel) + rgb_center_x + dx
    y = (y_e - event_center_y) * (rgb_pixel / event_pixel) + rgb_center_y + dy

    x = np.clip(x, 0, rgb_width - 1).astype(np.int32, copy=False)
    y = np.clip(y, 0, rgb_height - 1).astype(np.int32, copy=False)
    return x, y
# ---------------------------------------------------------------------

def clip_box(x, y, w, h, W, H):
    x = int(max(0, x)); y = int(max(0, y))
    w = int(max(0, w)); h = int(max(0, h))
    if x + w > W: w = max(0, W - x)
    if y + h > H: h = max(0, H - y)
    return x, y, w, h

def transform_points_to_voxels(data_dict, voxel_generator, device, save_voxel=5_000):
    pts = data_dict['points'].to(device)  # (N,4) = (t,x,y,p)

    if pts.numel() == 0:
        # Return empty tensors with correct shapes
        return (torch.empty((0,16,4), device=device),
                torch.empty((0,3), dtype=torch.int32, device=device))

    # Randomize to avoid bias toward early points
    shuffle = torch.randperm(pts.shape[0], device=device)
    pts = pts[shuffle]

    voxels, coords, num_pts = voxel_generator(pts)

    if num_pts.numel() == 0:
        return (torch.empty((0,16,4), device=device),
                torch.empty((0,3), dtype=torch.int32, device=device))

    # keep densest voxels
    if num_pts.shape[0] > save_voxel:
        _, idx = torch.topk(num_pts, save_voxel)
        voxels, coords, num_pts = voxels[idx], coords[idx], num_pts[idx]

    # Your pipeline expects (t,x,y). spconv usually returns (z,y,x)=(t,y,x)
    coords = coords.to(device)
    coords[:, [0, 1, 2]] = coords[:, [2, 1, 0]]  # (t,x,y)

    feats = voxels[:, :, 3]  # polarity channel
    return coords, feats

def align_buffer(root_dir, raw_data_dir, index,
                 bias=0, skip_start=0, skip_end=0,
                 debug_show=False, save_voxel=5000,
                 eps=12, min_samples=60,
                 bbox_range=10, event_thresh=40):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_core.event_io import EventsIterator

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load event packets (DEEP COPY each chunk) ---
    ev_path = f"{raw_data_dir}/event/{index}.raw"
    event_iterator = iter(EventsIterator(
        input_path=ev_path,
        mode="delta_t",
        delta_t=2000,
        n_events=1_000_000
    ))

    event_buffer = []
    for ev in event_iterator:
        # deep copy the structured array FIRST
        ev_local = np.empty(ev.shape, dtype=ev.dtype)
        np.copyto(ev_local, ev)

        # rotate/translate on the copy
        ev_local['x'], ev_local['y'] = ev_local['y'].copy(), -ev_local['x'].copy()
        ev_local['y'] += 1279
        ev_local['x'], ev_local['y'] = align_to_rgb(ev_local)

        event_buffer.append(ev_local)

    event_count = len(event_buffer)
    if event_count == 0:
        print(f"[{index}] No event packets found.")
        return

    # --- Load RGB frames (buffered; for very long videos, stream instead) ---
    vid_path = f"{raw_data_dir}/rgb/{index}.mp4"
    cap = cv2.VideoCapture(vid_path)
    rgb_buffer = []
    ok, frame = cap.read()
    while ok:
        rgb_buffer.append(frame)
        ok, frame = cap.read()
    cap.release()

    rgb_count = len(rgb_buffer)
    if rgb_count == 0:
        print(f"[{index}] No RGB frames found.")
        return

    print(f"[{index}] total rgb frames: {rgb_count}")
    print(f"[{index}] total event frames: {event_count}")
    print("align start")

    # --- Prepare output dirs ---
    out_dir = f"{root_dir}/{index}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(f"{out_dir}/rgb")
    os.makedirs(f"{out_dir}/event")
    os.makedirs(f"{out_dir}/combined")
    os.makedirs(f"{out_dir}/voxel")

    # --- Event frame generator & clustering ---
    rgb_w, rgb_h = 1080, 1440
    frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=rgb_w, sensor_height=rgb_h,
                                                 fps=100, palette=ColorPalette.Dark)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # --- Voxel generator: ranges must match your scaled (t,x,y) and image size ---
    voxel_gen = PointToVoxel(
        vsize_xyz=[50, 10, 10],
        coors_range_xyz=[0, 0, 0, 1000, rgb_w, rgb_h],  # t in [0,1000], x in [0,1080], y in [0,1440]
        num_point_features=4,
        max_num_voxels=16000,
        max_num_points_per_voxel=16,
        device=device
    )

    # --- Time alignment ---
    denom = max(rgb_count - 1, 1)
    ratio = max((event_count - bias) / denom, 0.0)

    absent_list, bbox = [], []

    start_idx = max(0, int(skip_start))
    end_idx = rgb_count - max(0, int(skip_end))
    end_idx = max(end_idx, start_idx)

    for frame_idx in range(start_idx, end_idx):
        if frame_idx % 50 == 0:
            print(f"[{index}] processing frame {frame_idx}/{end_idx-1}")

        rgb_img = rgb_buffer[frame_idx]
        if rgb_img is None:
            absent_list.append(0)
            bbox.append([0, 0, 0, 0])
            continue

        # Corresponding event packet (clamped)
        ev_idx = int(frame_idx * ratio) + bias
        ev_idx = min(max(ev_idx, 0), event_count - 1)
        ev_local = event_buffer[ev_idx]

        # --- Event image ---
        event_frame = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)
        frame_gen.generate_frame(ev_local, event_frame)
        gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, event_thresh, 255, cv2.THRESH_BINARY)

        points = np.column_stack(np.where(binary > 0))  # (row=y, col=x)
        if points.shape[0] >= min_samples:
            dbscan.fit(points)
            labels = dbscan.labels_
            sel = labels != -1
            cluster_points = points[sel] if sel.any() else np.empty((0, 2), dtype=np.int32)
        else:
            cluster_points = np.empty((0, 2), dtype=np.int32)

        event_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # --- RGB mask (green) ---
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([30, 90, 50], dtype=np.uint8)
        upper_hsv = np.array([50, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Fuse bboxes ---
        present = 0
        x = y = w = h = 0

        # Candidate from RGB
        rgb_box = None
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            rgb_box = cv2.boundingRect(c)  # (x,y,w,h)

        # Candidate from events
        ev_box = None
        if cluster_points.shape[0] > 0:
            y_min, x_min = int(cluster_points[:, 0].min()), int(cluster_points[:, 1].min())
            y_max, x_max = int(cluster_points[:, 0].max()), int(cluster_points[:, 1].max())
            ev_box = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Prefer union if both exist
        if rgb_box and ev_box:
            x1 = min(rgb_box[0], ev_box[0])
            y1 = min(rgb_box[1], ev_box[1])
            x2 = max(rgb_box[0] + rgb_box[2], ev_box[0] + ev_box[2])
            y2 = max(rgb_box[1] + rgb_box[3], ev_box[1] + ev_box[3])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            present = 1
        elif rgb_box:
            x, y, w, h = rgb_box
            present = 1
        elif ev_box:
            x, y, w, h = ev_box
            present = 1
        else:
            present = 0
            x = y = w = h = 0

        # Reject outliers / off-image
        if present and (w > 100 or h > 100 or y >= rgb_h):
            present = 0
            x = y = w = h = 0

        # Expand and clip
        if present:
            x -= bbox_range
            y -= bbox_range
            w += 2 * bbox_range
            h += 2 * bbox_range
            x, y, w, h = clip_box(x, y, w, h, rgb_w, rgb_h)

        # Draw (no GUI; just onto saved images)
        # if present and w > 0 and h > 0:
        #     cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.rectangle(event_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        absent_list.append(1 if present else 0)
        bbox.append([x, y, w, h])
        # alpha = 0.7  # 0 = 全透明（看不到前景），1 = 不透明
        #
        # # combined = cv2.addWeighted(event_image, alpha, rgb_image, 1 - alpha, 0)
        # combined = np.hstack((rgb_img, event_vis))
        # cv2.imshow('combined', combined)
        # cv2.waitKey(1)
        # --- Save images ---
        out_idx = frame_idx - start_idx
        cv2.imwrite(f"{out_dir}/rgb/{out_idx}.jpg", rgb_img)
        cv2.imwrite(f"{out_dir}/event/{out_idx}.jpg", event_vis)
        cv2.imwrite(f"{out_dir}/combined/{out_idx}.jpg", np.hstack((rgb_img, event_vis)))

        # --- Voxelization (use copies when converting to tensors) ---
        if ev_local['t'].shape[0] > 0:
            arr_t = ev_local["t"].astype(np.float32, copy=True)
            t_event = torch.as_tensor(arr_t, device=device).reshape(-1, 1)

            x_event = torch.as_tensor(ev_local["x"].astype(np.int32, copy=True), device=device).reshape(-1, 1)
            y_event = torch.as_tensor(ev_local["y"].astype(np.int32, copy=True), device=device).reshape(-1, 1)
            p_arr = ev_local["p"].astype(np.int8, copy=True)
            p_arr[p_arr == 0] = -1
            p_event = torch.as_tensor(p_arr, device=device).reshape(-1, 1)

            dt = float(arr_t[-1] - arr_t[0])
            if dt > 0:
                t_scaled = (t_event - t_event[0]) / dt * 1000.0
            else:
                t_scaled = torch.zeros_like(t_event)

            event_points = torch.cat((t_scaled, x_event, y_event, p_event), dim=1)  # (N,4) = (t,x,y,p)

            coor, feats = transform_points_to_voxels(
                {'points': event_points},
                voxel_generator=voxel_gen,
                device=device,
                save_voxel=save_voxel
            )

            sio.savemat(os.path.join(f"{out_dir}/voxel", f"{out_idx}.mat"),
                        {"coor": coor.detach().cpu().numpy(),
                         "features": feats.detach().cpu().numpy()})

    print("align finished")

    # --- Write meta and copies ---
    shutil.copy(ev_path, f"{out_dir}/event.raw")

    with open(f"{out_dir}/absent.txt", 'w') as f:
        for a in absent_list:
            f.write(f"{a}\n")

    with open(f"{out_dir}/groundtruth.txt", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(bbox)

    with open(f"{out_dir}/adjust.txt", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([bias, skip_start, skip_end])

def main():
    root_dir = "/media/xinchi/data/Event_RGB_catching_data/dataset"
    raw_data_dir = "/media/xinchi/data/Event_RGB_catching_data/saved_data_d"

    for index in range(0, 100):
        align_buffer(root_dir, raw_data_dir, index,
                     bias=0, skip_start=0, skip_end=0,
                     debug_show=True,  # keep False for servers/headless
                     save_voxel=5000, eps=12, min_samples=60,
                     bbox_range=10, event_thresh=40)

if __name__ == "__main__":
    main()
