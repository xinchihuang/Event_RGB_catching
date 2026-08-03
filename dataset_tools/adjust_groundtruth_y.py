#!/usr/bin/env python3
"""
Shift the y coordinate in all groundtruth.txt files under a dataset root.

Usage:
  python3 dataset_tools/adjust_groundtruth_y.py
  (Edit dataset_path and y_offset in main() to suit your dataset.)
"""
import os
import csv

def adjust_groundtruth_y(dataset_path, y_offset=-30):
    """
    Adjust y values in all groundtruth.txt files.
    Args:
        dataset_path: dataset root path
        y_offset: y offset to add (default: -30)
    """
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return
    
    # Collect all sequence folders.
    sequence_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            sequence_dirs.append(int(item))
    
    sequence_dirs.sort()
    print(f"Found {len(sequence_dirs)} sequence directories")
    
    total_processed = 0
    total_modified = 0
    
    for seq_id in sequence_dirs:
        seq_path = os.path.join(dataset_path, str(seq_id))
        groundtruth_file = os.path.join(seq_path, "groundtruth.txt")
        
        if not os.path.exists(groundtruth_file):
            print(f"Skipping sequence {seq_id}: no groundtruth.txt found")
            continue
        
        print(f"Processing sequence {seq_id}...")
        
        # Read existing data.
        bbox_data = []
        with open(groundtruth_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        x, y, w, h = map(int, parts[:4])
                        bbox_data.append([x, y, w, h])
                    except ValueError:
                        print(f"Warning: Invalid data in {groundtruth_file}: {line}")
                        bbox_data.append([0, 0, 0, 0])
                else:
                    print(f"Warning: Incomplete data in {groundtruth_file}: {line}")
                    bbox_data.append([0, 0, 0, 0])
        
        # Apply y shift.
        modified_count = 0
        for bbox in bbox_data:
            x, y, w, h = bbox
            if y > 0:  # Only adjust non-zero entries.
                new_y = max(0, y + y_offset)  # Keep y >= 0.
                if new_y != y:
                    bbox[1] = new_y
                    modified_count += 1
        
        # Write back to file.
        with open(groundtruth_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for bbox in bbox_data:
                writer.writerow(bbox)
        
        print(f"  Processed {len(bbox_data)} bounding boxes, modified {modified_count}")
        total_processed += len(bbox_data)
        total_modified += modified_count
    
    print(f"\nSummary:")
    print(f"  Total sequences processed: {len(sequence_dirs)}")
    print(f"  Total bounding boxes processed: {total_processed}")
    print(f"  Total bounding boxes modified: {total_modified}")
    print(f"  Y offset applied: {y_offset}")

def main():
    dataset_path = "/media/xinchi/data/saved_data/selected_dataset"
    y_offset = -30
    
    print("Adjusting groundtruth.txt files...")
    print(f"Dataset path: {dataset_path}")
    print(f"Y offset: {y_offset}")
    print()
    
    # Confirm destructive update.
    response = input("This will modify all groundtruth.txt files. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    adjust_groundtruth_y(dataset_path, y_offset)
    print("Done!")

if __name__ == "__main__":
    main()
