#!/usr/bin/env python3
import os

def load_frame_flags(dataset_path):
    """
    从frame_flags.txt加载标记数据
    返回格式: {seq_id: {frame_id: flag}}
    """
    flags_file = os.path.join(dataset_path, "frame_flags.txt")
    frame_flags = {}
    
    if not os.path.exists(flags_file):
        print(f"Frame flags file not found: {flags_file}")
        return frame_flags
    
    with open(flags_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) != 3:
                print(f"Warning: Invalid format in line {line_num}: {line}")
                continue
                
            try:
                seq_id, frame_id, flag = int(parts[0]), int(parts[1]), int(parts[2])
                if seq_id not in frame_flags:
                    frame_flags[seq_id] = {}
                frame_flags[seq_id][frame_id] = bool(flag)
            except ValueError:
                print(f"Warning: Invalid data in line {line_num}: {line}")
                continue
    
    return frame_flags

def update_absent_files(dataset_path, frame_flags):
    """
    根据frame_flags更新absent.txt文件
    逻辑：
    1. 如果帧被明确标记为flag=1，设为present (1)
    2. 如果帧被明确标记为flag=0，设为absent (0) 
    3. 如果帧没有被标记，设为absent (0)
    """
    
    if not frame_flags:
        print("No frame flags found, nothing to update.")
        return 0, 0
    
    # 获取所有序列
    all_sequences = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            all_sequences.append(int(item))
    
    all_sequences.sort()
    total_sequences = 0
    total_modified = 0
    
    for seq_id in all_sequences:
        seq_path = os.path.join(dataset_path, str(seq_id))
        absent_file = os.path.join(seq_path, "absent.txt")
        
        if not os.path.exists(absent_file):
            print(f"Warning: absent.txt not found for sequence {seq_id}")
            continue
        
        print(f"Processing sequence {seq_id}...")
        
        # 读取absent.txt
        absent_data = []
        with open(absent_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        absent_data.append(int(line))
                    except ValueError:
                        print(f"Warning: Invalid data in absent.txt for sequence {seq_id}: {line}")
                        absent_data.append(0)
        
        modified_frames = 0
        
        # 获取该序列的标记信息
        seq_flags = frame_flags.get(seq_id, {})
        
        # 遍历所有帧
        for frame_id in range(len(absent_data)):
            if frame_id in seq_flags:
                # 帧有明确标记
                should_be_present = seq_flags[frame_id]
                new_value = 1 if should_be_present else 0
            else:
                # 帧没有标记，默认设为absent
                new_value = 0
            
            if absent_data[frame_id] != new_value:
                old_value = absent_data[frame_id]
                absent_data[frame_id] = new_value
                modified_frames += 1
                status = "present" if new_value == 1 else "absent"
                old_status = "present" if old_value == 1 else "absent"
                flag_status = "flagged" if frame_id in seq_flags else "not flagged"
                print(f"  Frame {frame_id}: set to {status} (was {old_status}, {flag_status})")
        
        # 写回absent.txt
        if modified_frames > 0:
            with open(absent_file, 'w') as f:
                for absent_value in absent_data:
                    f.write(f"{absent_value}\n")
            
            print(f"  Modified {modified_frames} frames in sequence {seq_id}")
            total_modified += modified_frames
        else:
            print(f"  No changes needed for sequence {seq_id}")
        
        total_sequences += 1
    
    return total_sequences, total_modified

def main():
    dataset_path = "/media/xinchi/data/saved_data/selected_dataset"
    
    print("Updating absent.txt files based on frame_flags.txt...")
    print(f"Dataset path: {dataset_path}")
    print()
    
    # 加载frame flags
    print("Loading frame flags...")
    frame_flags = load_frame_flags(dataset_path)
    
    if not frame_flags:
        print("No frame flags found. Exiting.")
        return
    
    # 获取所有序列以计算统计信息
    all_sequences = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            all_sequences.append(int(item))
    
    # 计算统计信息
    total_sequences_with_flags = len(frame_flags)
    total_flagged_frames = sum(len(frames) for frames in frame_flags.values())
    flagged_as_present = sum(1 for frames in frame_flags.values() 
                            for flag in frames.values() if flag)
    flagged_as_absent = sum(1 for frames in frame_flags.values() 
                           for flag in frames.values() if not flag)
    
    # 计算总帧数
    total_frames = 0
    for seq_id in all_sequences:
        absent_file = os.path.join(dataset_path, str(seq_id), "absent.txt")
        if os.path.exists(absent_file):
            with open(absent_file, 'r') as f:
                total_frames += len([line for line in f if line.strip()])
    
    unflagged_frames = total_frames - total_flagged_frames
    
    print(f"Found flags for {total_sequences_with_flags} sequences")
    print(f"Total sequences in dataset: {len(all_sequences)}")
    print(f"Total frames in dataset: {total_frames}")
    print(f"Flagged frames: {total_flagged_frames}")
    print(f"  - Flagged as present (flag=1): {flagged_as_present}")
    print(f"  - Flagged as absent (flag=0): {flagged_as_absent}")
    print(f"Unflagged frames (will be set to absent): {unflagged_frames}")
    print()
    
    # 确认操作
    frames_to_modify = flagged_as_absent + unflagged_frames
    if frames_to_modify == 0:
        print("No frames need to be modified. Exiting.")
        return
        
    response = input(f"This will modify absent.txt files for {frames_to_modify} frames. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # 执行更新
    print("\nUpdating absent.txt files...")
    total_sequences, total_modified = update_absent_files(dataset_path, frame_flags)
    
    print(f"\nSummary:")
    print(f"  Sequences processed: {total_sequences}")
    print(f"  Total frames modified: {total_modified}")
    print("Done!")

if __name__ == "__main__":
    main()