"""
Generate dataset frames (RGB/event/combined/voxel) from raw recordings.

Usage:
  python3 dataset_tools/prepare_dataset.py
  --task normal --record-dir /path/to/raw --output-dir /path/to/dataset --index 13 --count 87
"""
import os
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import SCENES, stage_path, resolve_path, require_recordings, distinct_output
import csv
import cv2
import time
import shutil
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import scipy.io as sio

# ---------------------------------------------------------------------
# Optional: if you already have align_to_rgb elsewhere, you can remove this.
# This version matches your commented code (with tweakable offsets).
def align_to_rgb(event_data,rgb_width=1080, rgb_height = 1440, event_width=720, event_height = 1280):
    x_e,y_e=event_data['x'].copy(),event_data['y'].copy()
    rgb_center_x=rgb_width/2
    rgb_center_y=rgb_height/2
    # print(rgb_center_x,rgb_center_y)
    event_center_x=event_width/2
    event_center_y=event_height/2
    event_pixel=3.45
    rgb_pixel=4.86

    x = (x_e-event_center_x) * rgb_pixel / event_pixel + rgb_center_x+30
    y = (y_e-event_center_y) * rgb_pixel / event_pixel + rgb_center_y+29
    y= np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)

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
    from spconv.pytorch.utils import PointToVoxel
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_core.event_io import EventsIterator
    import gc

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
    try:
        for ev in event_iterator:
            if ev is None or ev.size == 0:
                continue
            
            # deep copy the structured array FIRST with validation
            try:
                ev_local = np.empty(ev.shape, dtype=ev.dtype)
                np.copyto(ev_local, ev)
            except Exception as e:
                print(f"Warning: failed to copy event data: {e}")
                continue

            # rotate/translate on the copy
            try:
                ev_local['x'], ev_local['y'] = ev_local['y'].copy(), -ev_local['x'].copy()
                ev_local['y'] += 1279
                ev_local['x'], ev_local['y'] = align_to_rgb(ev_local)
            except Exception as e:
                print(f"Warning: coordinate transformation failed: {e}")
                continue

            event_buffer.append(ev_local)
            
            # Periodic garbage collection to prevent memory buildup
            if len(event_buffer) % 100 == 0:
                gc.collect()
                
    except Exception as e:
        print(f"Error loading event buffer: {e}")
        if len(event_buffer) == 0:
            return

    event_count = len(event_buffer)
    if event_count == 0:
        print(f"[{index}] No event packets found.")
        return

    # --- Load RGB frames (buffered; for very long videos, stream instead) ---
    vid_path = f"{raw_data_dir}/rgb/{index}.mp4"
    print(vid_path)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"[{index}] Cannot open video file: {vid_path}")
        return
        
    rgb_buffer = []
    try:
        ok, frame = cap.read()
        while ok:
            if frame is not None:
                # Create explicit copy to avoid shared memory issues
                rgb_buffer.append(frame.copy())
            ok, frame = cap.read()
            
            # Periodic garbage collection
            if len(rgb_buffer) % 100 == 0:
                gc.collect()
    except Exception as e:
        print(f"Error reading RGB frames: {e}")
    finally:
        cap.release()



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
    rgb_count = len(rgb_buffer)
    if rgb_count == 0:
        print(f"[{index}] No RGB frames found.")
        return

    print(f"[{index}] total rgb frames: {rgb_count}")
    print(f"[{index}] total event frames: {event_count}")
    print("align start")
    # --- Time alignment ---
    ratio=10
    absent_list, bbox = [], []
    rgb_bias, event_bias = 0, 0
    if rgb_count*10>event_count:
        rgb_bias=int(rgb_count-event_count/10)
        rgb_buffer=rgb_buffer[:-rgb_bias]
    else:
        event_bias=int(event_count-rgb_count*10)
        event_buffer=event_buffer[event_bias:]

    print(rgb_bias,event_bias)
    start_idx=0
    end_idx=len(rgb_buffer)
    for frame_idx in range(0, end_idx):
        try:
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
            
            # Validate event data
            if ev_local is None:
                print(f"Warning: None event data for frame {frame_idx}")
                absent_list.append(0)
                bbox.append([0, 0, 0, 0])
                continue

            # --- Event image ---
            event_frame = None
            try:
                event_frame = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)
                
                # Check if ev_local has events before processing
                if ev_local is not None and len(ev_local) > 0:
                    # Validate event data bounds to prevent buffer overflow
                    if np.any(ev_local['x'] < 0) or np.any(ev_local['x'] >= rgb_w) or \
                       np.any(ev_local['y'] < 0) or np.any(ev_local['y'] >= rgb_h):
                        # Clip coordinates to prevent buffer overflow
                        ev_local['x'] = np.clip(ev_local['x'], 0, rgb_w - 1)
                        ev_local['y'] = np.clip(ev_local['y'], 0, rgb_h - 1)
                    
                    try:
                        frame_gen.generate_frame(ev_local, event_frame)
                    except Exception as e:
                        print(f"Warning: frame generation failed for frame {frame_idx}, ev_idx {ev_idx}: {e}")
                        # Create fresh event frame to avoid corruption
                        event_frame = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)
                        
            except Exception as e:
                print(f"Warning: event frame creation failed for frame {frame_idx}: {e}")
                event_frame = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)
            
            gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, event_thresh, 255, cv2.THRESH_BINARY)

            points = np.column_stack(np.where(binary > 0))  # (row=y, col=x)
            
            # Apply filter for bounding box detection only (exclude events within specified range)
            if points.shape[0] > 0:
                # Convert points to x,y format and apply filter
                points_x, points_y = points[:, 1], points[:, 0]  # col=x, row=y
                valid_mask = ~((points_x >= 600) & (points_x < 1080) & (points_y >= 900) & (points_y < 1440))
                points = points[valid_mask]
            
            if points.shape[0] >= min_samples:
                dbscan.fit(points)
                labels = dbscan.labels_
                sel = labels != -1
                cluster_points = points[sel] if np.any(sel) else np.empty((0, 2), dtype=np.int32)
            else:
                cluster_points = np.empty((0, 2), dtype=np.int32)

            # Create two versions: filtered for display, unfiltered for saving
            # Unfiltered version for saving
            event_vis_save = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # Filtered version for display (exclude events within specified range)
            binary_filtered = binary.copy()
            if binary_filtered.shape == binary.shape:  # Ensure shapes match
                try:
                    y_coords, x_coords = np.where(binary > 0)
                    # Validate coordinates are within bounds before using them
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        valid_indices = (y_coords >= 0) & (y_coords < binary.shape[0]) & \
                                       (x_coords >= 0) & (x_coords < binary.shape[1])
                        y_coords = y_coords[valid_indices]
                        x_coords = x_coords[valid_indices]
                        
                        if len(y_coords) > 0:  # Only proceed if we have valid coordinates
                            filter_mask = (x_coords >= 600) & (x_coords < 1080) & (y_coords >= 900) & (y_coords < 1440)
                            if np.any(filter_mask):
                                binary_filtered[y_coords[filter_mask], x_coords[filter_mask]] = 0
                except Exception as e:
                    print(f"Warning: filtering failed for frame {frame_idx}: {e}")
                    
            event_vis_display = cv2.cvtColor(binary_filtered, cv2.COLOR_GRAY2BGR)

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

            # Create RGB+Event overlay display similar to prepare_dataset_multiprocess.py
            rgb_brightened = cv2.convertScaleAbs(rgb_img, alpha=0.7, beta=30)
            overlay = cv2.addWeighted(rgb_brightened, 0.7, event_vis_display, 0.3, 0)
            
            # Draw bounding box on overlay
            if present and w > 0 and h > 0:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Add bbox info text
                cv2.putText(overlay, f"Ball: {w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add informative text overlay
            cv2.putText(overlay, f"Record: {index} Frame: {frame_idx}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(overlay, f"Events: {len(ev_local)}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(overlay, f"Ball: {'Present' if present else 'Absent'}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if present else (0, 0, 255), 2)
            
            # Add frame indices for debugging
            cv2.putText(overlay, f"RGB Frame: {frame_idx}", (20, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(overlay, f"Event Frame: {ev_idx}", (20, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Scale for better display
            scale = 0.8
            display_width = int(overlay.shape[1] * scale)
            display_height = int(overlay.shape[0] * scale)
            display_img = cv2.resize(overlay, (display_width, display_height), interpolation=cv2.INTER_AREA)
            
            # Display the overlay
            cv2.imshow(f'Dataset Processing - Record {index}', display_img)
            cv2.waitKey(1)

            absent_list.append(1 if present else 0)
            bbox.append([x, y, w, h])
            
            # Also create combined view for saving (side-by-side with bounding boxes)
            rgb_img_show = rgb_img.copy()
            event_vis_show = event_vis_save.copy()
            if present and w > 0 and h > 0:
                cv2.rectangle(rgb_img_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(event_vis_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
            combined_sidebyside = np.hstack((rgb_img_show, event_vis_show))
            # --- Save images ---
            out_idx = frame_idx - start_idx
            cv2.imwrite(f"{out_dir}/rgb/{out_idx}.jpg", rgb_img)
            cv2.imwrite(f"{out_dir}/event/{out_idx}.jpg", event_vis_save)
            cv2.imwrite(f"{out_dir}/combined/{out_idx}.jpg", np.hstack((rgb_img, event_vis_save)))

            # --- Voxelization (use copies when converting to tensors) ---
            if ev_local is not None and ev_local.shape[0] > 0 and 't' in ev_local.dtype.names:
                t_event = x_event = y_event = p_event = event_points = None
                try:
                    # Validate event data size before processing
                    if ev_local.shape[0] > 1_000_000:  # Prevent memory explosion
                        print(f"Warning: too many events ({ev_local.shape[0]}) for frame {frame_idx}, sampling...")
                        indices = np.random.choice(ev_local.shape[0], 1_000_000, replace=False)
                        ev_local_sampled = ev_local[indices]
                    else:
                        ev_local_sampled = ev_local
                    
                    arr_t = ev_local_sampled["t"].astype(np.float32, copy=True)
                    
                    # Check for valid time data
                    if arr_t.shape[0] == 0:
                        print(f"Warning: empty time array for frame {frame_idx}")
                        continue
                    
                    t_event = torch.as_tensor(arr_t, device=device).reshape(-1, 1)

                    x_event = torch.as_tensor(ev_local_sampled["x"].astype(np.int32, copy=True), device=device).reshape(-1, 1)
                    y_event = torch.as_tensor(ev_local_sampled["y"].astype(np.int32, copy=True), device=device).reshape(-1, 1)
                    p_arr = ev_local_sampled["p"].astype(np.int8, copy=True)
                    p_arr[p_arr == 0] = -1
                    p_event = torch.as_tensor(p_arr, device=device).reshape(-1, 1)

                    # Validate data consistency
                    if not (t_event.shape[0] == x_event.shape[0] == y_event.shape[0] == p_event.shape[0]):
                        print(f"Warning: inconsistent event data shapes for frame {frame_idx}")
                        continue

                    dt = float(arr_t[-1] - arr_t[0]) if arr_t.shape[0] > 1 else 1.0
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

                    # Check if voxelization produced valid results
                    if coor.numel() > 0 and feats.numel() > 0:
                        sio.savemat(os.path.join(f"{out_dir}/voxel", f"{out_idx}.mat"),
                                    {"coor": coor.detach().cpu().numpy(),
                                     "features": feats.detach().cpu().numpy()})
                    else:
                        print(f"Warning: empty voxel data for frame {frame_idx}")
                        # Create empty voxel file to maintain sequence
                        sio.savemat(os.path.join(f"{out_dir}/voxel", f"{out_idx}.mat"),
                                    {"coor": np.empty((0, 3)), "features": np.empty((0, 16))})
                        
                except Exception as e:
                    print(f"Warning: voxelization failed for frame {frame_idx}: {e}")
                    # Create empty voxel file to maintain sequence
                    try:
                        sio.savemat(os.path.join(f"{out_dir}/voxel", f"{out_idx}.mat"),
                                    {"coor": np.empty((0, 3)), "features": np.empty((0, 16))})
                    except Exception as save_e:
                        print(f"Warning: could not save empty voxel file for frame {frame_idx}: {save_e}")
                        
                finally:
                    # Explicit cleanup to prevent memory leaks
                    try:
                        if 't_event' in locals() and t_event is not None:
                            del t_event
                        if 'x_event' in locals() and x_event is not None:
                            del x_event
                        if 'y_event' in locals() and y_event is not None:
                            del y_event
                        if 'p_event' in locals() and p_event is not None:
                            del p_event
                        if 'event_points' in locals() and event_points is not None:
                            del event_points
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    except:
                        pass
        
            # Periodic cleanup every 50 frames
            if frame_idx % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Still append to lists to maintain sequence
            absent_list.append(0)
            bbox.append([0, 0, 0, 0])
            
            # Force cleanup on error
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            continue

    print("align finished")
    
    # Close display windows after processing this record
    cv2.destroyAllWindows()
    
    # Final cleanup
    try:
        del event_buffer
        del rgb_buffer
        del frame_gen
        del dbscan
        del voxel_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

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

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=SCENES, default='normal')
    parser.add_argument('--record-dir', help='Raw recordings, before time_align_rewrite')
    parser.add_argument('--output-dir', help='Training dataset output')
    parser.add_argument('--index', type=int, default=13, help='First recording ID (default: 13)')
    parser.add_argument('--count', type=int, default=87, help='Number of IDs (default: 87)')
    args = parser.parse_args(argv)
    if args.index < 0 or args.count <= 0:
        parser.error('index must be non-negative and count must be positive')
    try:
        args.record_dir = str(stage_path('raw', args.task, args.record_dir))
        args.output_dir = str(distinct_output(args.record_dir,
            stage_path('dataset', args.task, args.output_dir)))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    require_recordings(args.record_dir)
    print(f"Raw input: {args.record_dir}\nTraining output: {args.output_dir}", flush=True)
    for index in range(args.index, args.index + args.count):
        for relative in (f'event/{index}.raw', f'rgb/{index}.mp4'):
            path = Path(args.record_dir) / relative
            if not path.is_file():
                raise FileNotFoundError(f'Missing raw input: {path}')
        align_buffer(args.output_dir, args.record_dir, index,
                     bias=-15, skip_start=0, skip_end=0,
                     debug_show=True,
                     save_voxel=5000, eps=8, min_samples=60,
                     bbox_range=30, event_thresh=40)

if __name__ == "__main__":
    main()
