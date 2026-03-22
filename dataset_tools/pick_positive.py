"""
Build a merged dataset by copying all positive samples and a random subset of negatives.

Usage:
  python3 dataset_tools/pick_positive.py
  (Edit source_datasets, negative_sample_probability, and output_dir in this file.)
"""
import os
import shutil
import random
import cv2
import numpy as np

# Source dataset roots to merge.
source_datasets = [
    "/media/xinchi/data/saved_data/dataset_c",
    "/media/xinchi/data/saved_data/dataset_d",
    "/media/xinchi/data/saved_data/dataset_dc",
    "/media/xinchi/data/saved_data/dataset_l",
    # Add more dataset roots here if needed.
]

# Probability to keep a negative sample (0.0-1.0).
negative_sample_probability = 0.1

output_dir = "/media/xinchi/data/saved_data/selected_dataset"
os.makedirs(output_dir, exist_ok=True)

# Track generated samples before writing train/list files.
train_lines = []
list_lines = []

# Global counter for the merged dataset sequences.
global_sequence_count = 0

for source_dir in source_datasets:
    for i in range(0, 100):
        data_path = os.path.join(source_dir, f"{i}")
        
        # Skip missing sequences.
        if not os.path.exists(data_path):
            continue
            
        absent_path = os.path.join(data_path, "absent.txt")
        ground_truth_path = os.path.join(data_path, "groundtruth.txt")
        
        # Require both metadata files.
        if not (os.path.exists(absent_path) and os.path.exists(ground_truth_path)):
            continue
        
        new_data_path = os.path.join(output_dir, f"{global_sequence_count}")
        new_absent_path = os.path.join(new_data_path, "absent.txt")
        new_ground_truth_path = os.path.join(new_data_path, "groundtruth.txt")

        os.makedirs(new_data_path, exist_ok=True)
        os.makedirs(os.path.join(new_data_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(new_data_path, "event"), exist_ok=True)
        os.makedirs(os.path.join(new_data_path, "combined"), exist_ok=True)
        os.makedirs(os.path.join(new_data_path, "voxel"), exist_ok=True)

        image_count = 0
        positive_samples_found = False
        
        with open(absent_path, 'r') as absent_file, \
             open(ground_truth_path, 'r') as ground_truth_file, \
             open(new_absent_path, 'w') as new_absent_file, \
             open(new_ground_truth_path, 'w') as new_ground_truth_file:

            for line_count, (absent_line, gt_line) in enumerate(zip(absent_file, ground_truth_file), start=1):
                is_positive = absent_line.strip() != '0'
                
                # Always keep positives.
                if is_positive:
                    include_sample = True
                # Keep negatives with a fixed probability.
                else:
                    include_sample = random.random() < negative_sample_probability
                
                if not include_sample:
                    continue

                new_absent_file.write(absent_line)
                new_ground_truth_file.write(gt_line)

                # Copy matching frame files.
                for file_type, extension in [("rgb", "jpg"), ("event", "jpg"), ("combined", "jpg"), ("voxel", "mat")]:
                    src_file = os.path.join(data_path, file_type, f"{line_count}.{extension}")
                    dst_file = os.path.join(new_data_path, file_type, f"{image_count}.{extension}")
                    
                    if os.path.exists(src_file):
                        if file_type == "rgb" and "dataset_dc" in source_dir:
                            # Brighten dataset_dc RGB images for consistency.
                            img = cv2.imread(src_file)
                            if img is not None:

                                # Increase brightness (adjust factor if needed).
                                img_brightened = np.clip(img * 3, 0, 255).astype(np.uint8)
                                cv2.imwrite(dst_file, img_brightened)
                            else:
                                shutil.copy(src_file, dst_file)
                        else:
                            shutil.copy(src_file, dst_file)

                image_count += 1
                positive_samples_found = True

        # Increment only if any samples were kept.
        if positive_samples_found:
            global_sequence_count += 1
        else:
            # Remove empty sequence folders.
            shutil.rmtree(new_data_path, ignore_errors=True)

# Write train/list files based on the final sequence count.
for i in range(global_sequence_count):
    train_lines.append(f"{i}\n")
    list_lines.append(f"{i}\n")

# Persist index lists.
with open(os.path.join(output_dir, "train.txt"), 'w') as f:
    f.writelines(train_lines)

with open(os.path.join(output_dir, "list.txt"), 'w') as f:
    f.writelines(list_lines)
