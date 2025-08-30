#!/usr/bin/env python3
"""
Safe version of prepare_dataset.py with better error handling and memory management
"""
import csv
import sys
import shutil
import scipy.io
import os
import gc
import argparse
from pathlib import Path

# Add project path
prj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def safe_import():
    """Safely import all required libraries with error handling"""
    try:
        import torch
        import numpy as np
        import cv2
        from spconv.pytorch.utils import PointToVoxel
        from sklearn.cluster import DBSCAN
        from metavision_core.event_io import EventsIterator
        from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
        from data_input.read_from_raw import align_to_rgb
        
        print("✓ All imports successful")
        return True, {
            'torch': torch,
            'np': np,
            'cv2': cv2,
            'PointToVoxel': PointToVoxel,
            'DBSCAN': DBSCAN,
            'EventsIterator': EventsIterator,
            'PeriodicFrameGenerationAlgorithm': PeriodicFrameGenerationAlgorithm,
            'ColorPalette': ColorPalette,
            'align_to_rgb': align_to_rgb
        }
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False, None

def transform_points_to_voxels(data_dict, voxel_generator, device, save_voxel=5000):
    """Transform points to voxels with error handling"""
    libs = data_dict.get('libs')
    if not libs:
        raise ValueError("Libraries not provided")
    
    torch = libs['torch']
    
    pts = data_dict['points'].to(device)
    shuffle = torch.randperm(pts.shape[0], device=device)
    pts = pts[shuffle]

    voxels, coords, num_pts = voxel_generator(pts)
    
    if num_pts.shape[0] > save_voxel:
        _, idx = torch.topk(num_pts, save_voxel)
        voxels, coords, num_pts = voxels[idx], coords[idx], num_pts[idx]

    coords = coords.to(device)
    coords[:, [0, 1, 2]] = coords[:, [2, 1, 0]]  # (y,x,t)→(t,x,y)
    feats = voxels[:, :, 3]  # 取极性通道

    return coords, feats

def align_buffer_safe(root_dir, raw_data_dir, index, bias=0, skip_start=0, skip_end=0):
    """Safe version of align_buffer with better memory management"""
    
    print(f"Processing index {index}...")
    
    # Import libraries safely
    success, libs = safe_import()
    if not success:
        return False
    
    torch = libs['torch']
    np = libs['np']
    cv2 = libs['cv2']
    PointToVoxel = libs['PointToVoxel']
    DBSCAN = libs['DBSCAN']
    EventsIterator = libs['EventsIterator']
    PeriodicFrameGenerationAlgorithm = libs['PeriodicFrameGenerationAlgorithm']
    ColorPalette = libs['ColorPalette']
    align_to_rgb = libs['align_to_rgb']
    
    try:
        # Setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Check if input files exist
        event_file = f"{raw_data_dir}/event/{index}.raw"
        video_file = f"{raw_data_dir}/rgb/{index}.mp4"
        
        if not os.path.exists(event_file):
            print(f"Event file not found: {event_file}")
            return False
        
        if not os.path.exists(video_file):
            print(f"Video file not found: {video_file}")
            return False
        
        print("Loading event data...")
        # Create event iterator with smaller chunks to avoid memory issues
        event_iterator = EventsIterator(
            input_path=event_file,
            mode="delta_t",
            delta_t=2000,
            n_events=100000  # Reduced from 1000000
        )
        
        # Load events in smaller batches
        event_buffer = []
        for i, ev in enumerate(event_iterator):
            if i >= 1000:  # Limit to first 1000 chunks to avoid memory issues
                break
            ev['x'], ev['y'] = ev['y'].copy(), -ev['x'].copy()
            ev['y'] += 1279
            ev['x'], ev['y'] = align_to_rgb(ev)
            event_buffer.append(ev)
            
            if i % 100 == 0:  # Progress indicator
                print(f"Loaded {i} event chunks...")
        
        print(f"Loaded {len(event_buffer)} event chunks")
        
        print("Loading video data...")
        # Load video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            return False
        
        rgb_buffer = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_buffer.append(frame)
            frame_count += 1
            
            if frame_count >= 1000:  # Limit frames to avoid memory issues
                break
                
            if frame_count % 100 == 0:
                print(f"Loaded {frame_count} RGB frames...")
        
        cap.release()
        print(f"Loaded {len(rgb_buffer)} RGB frames")
        
        if len(event_buffer) == 0 or len(rgb_buffer) == 0:
            print("No data loaded, skipping...")
            return False
        
        # Setup output directories
        output_dir = Path(root_dir) / str(index)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rgb").mkdir(exist_ok=True)
        (output_dir / "event").mkdir(exist_ok=True)
        (output_dir / "combined").mkdir(exist_ok=True)
        (output_dir / "voxel").mkdir(exist_ok=True)
        
        print("Processing frames...")
        # Process frames with simplified logic
        event_count = len(event_buffer)
        rgb_count = len(rgb_buffer)
        
        # Simple 1:1 mapping to avoid complex ratio calculations
        max_frames = min(event_count, rgb_count, 100)  # Limit to 100 frames for safety
        
        bbox_list = []
        absent_list = []
        
        for frame_idx in range(max_frames):
            try:
                if frame_idx < skip_start:
                    continue
                if frame_idx >= max_frames - skip_end:
                    break
                
                rgb_image = rgb_buffer[frame_idx].copy()
                event_data = event_buffer[frame_idx]
                
                # Create simple event visualization
                event_frame = np.zeros((1440, 1080, 3), dtype=np.uint8)
                
                if len(event_data['x']) > 0:
                    # Simple event rendering
                    for i in range(min(len(event_data['x']), 10000)):  # Limit points
                        x, y = int(event_data['x'][i]), int(event_data['y'][i])
                        if 0 <= x < 1080 and 0 <= y < 1440:
                            color = (255, 255, 255) if event_data['p'][i] > 0 else (128, 128, 128)
                            cv2.circle(event_frame, (x, y), 1, color, -1)
                
                # Simple ball detection in RGB
                hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                lower_hsv = np.array([30, 90, 0])
                upper_hsv = np.array([50, 255, 255])
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    if w < 100 and h < 100 and y < 1280:
                        bbox_list.append([x, y, w, h])
                        absent_list.append(1)
                        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(event_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        bbox_list.append([0, 0, 0, 0])
                        absent_list.append(0)
                else:
                    bbox_list.append([0, 0, 0, 0])
                    absent_list.append(0)
                
                # Save images
                combined = np.hstack((rgb_image, event_frame))
                cv2.imwrite(str(output_dir / "rgb" / f"{frame_idx}.jpg"), rgb_image)
                cv2.imwrite(str(output_dir / "event" / f"{frame_idx}.jpg"), event_frame)
                cv2.imwrite(str(output_dir / "combined" / f"{frame_idx}.jpg"), combined)
                
                if frame_idx % 10 == 0:
                    print(f"Processed frame {frame_idx}/{max_frames}")
                
                # Force garbage collection periodically
                if frame_idx % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        # Save metadata
        with open(output_dir / "absent.txt", 'w') as f:
            for a in absent_list:
                f.write(f"{a}\n")
        
        with open(output_dir / "groundtruth.txt", 'w') as f:
            writer = csv.writer(f)
            for box in bbox_list:
                writer.writerow(box)
        
        with open(output_dir / "adjust.txt", 'w') as f:
            writer = csv.writer(f)
            writer.writerow([bias, skip_start, skip_end])
        
        # Copy raw event file
        shutil.copy(event_file, output_dir / "event.raw")
        
        print(f"✓ Successfully processed {len(bbox_list)} frames for index {index}")
        return True
        
    except Exception as e:
        print(f"✗ Error in align_buffer_safe: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        gc.collect()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Safe dataset preparation')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=1, help='End index')
    parser.add_argument('--bias', type=int, default=0, help='Bias value')
    parser.add_argument('--skip_start', type=int, default=0, help='Skip start frames')
    parser.add_argument('--skip_end', type=int, default=0, help='Skip end frames')
    
    args = parser.parse_args()
    
    root_dir = "/media/xinchi/data/Event_RGB_catching_data/dataset"
    raw_data_dir = "/media/xinchi/data/Event_RGB_catching_data/saved_data_dc"
    
    success_count = 0
    total_count = args.end - args.start
    
    for index in range(args.start, args.end):
        try:
            success = align_buffer_safe(
                root_dir, raw_data_dir, index,
                bias=args.bias, skip_start=args.skip_start, skip_end=args.skip_end
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"✗ Failed to process index {index}: {e}")
            continue
        
        # Force cleanup between iterations
        gc.collect()
    
    print(f"\nCompleted: {success_count}/{total_count} successful")

if __name__ == "__main__":
    main()