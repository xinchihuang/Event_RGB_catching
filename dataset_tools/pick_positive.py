import os
import shutil
import random

# 配置多个源数据集路径
source_datasets = [
    # "/media/xinchi/data/Event_RGB_catching_data/dataset_l",
    # "/media/xinchi/data/Event_RGB_catching_data/dataset_d",
    "/media/xinchi/data/Event_RGB_catching_data/dataset_dc",
    # 在这里添加更多源数据集路径
]

# 配置负样本包含概率（0.0-1.0）
negative_sample_probability = 0.3

output_dir = "/media/xinchi/data/Event_RGB_catching_data/selected_dataset"
os.makedirs(output_dir, exist_ok=True)

# 先处理数据，统计实际生成的样本数量
train_lines = []
list_lines = []

# 全局计数器，用于合并后的数据集
global_sequence_count = 0

for source_dir in source_datasets:
    for i in range(0, 100):
        data_path = os.path.join(source_dir, f"{i}")
        
        # 检查数据路径是否存在
        if not os.path.exists(data_path):
            continue
            
        absent_path = os.path.join(data_path, "absent.txt")
        ground_truth_path = os.path.join(data_path, "groundtruth.txt")
        
        # 检查必要文件是否存在
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
                
                # 包含所有正样本
                if is_positive:
                    include_sample = True
                # 按概率包含负样本
                else:
                    include_sample = random.random() < negative_sample_probability
                
                if not include_sample:
                    continue

                new_absent_file.write(absent_line)
                new_ground_truth_file.write(gt_line)

                # 复制对应的文件
                for file_type, extension in [("rgb", "jpg"), ("event", "jpg"), ("combined", "jpg"), ("voxel", "mat")]:
                    src_file = os.path.join(data_path, file_type, f"{line_count}.{extension}")
                    dst_file = os.path.join(new_data_path, file_type, f"{image_count}.{extension}")
                    
                    if os.path.exists(src_file):
                        shutil.copy(src_file, dst_file)

                image_count += 1
                positive_samples_found = True

        # 如果找到了样本（正样本或负样本），则递增全局序列计数器
        if positive_samples_found:
            global_sequence_count += 1
        else:
            # 如果没有找到任何样本，删除创建的空文件夹
            shutil.rmtree(new_data_path, ignore_errors=True)

# 根据实际生成的样本数量生成train.txt和list.txt
for i in range(global_sequence_count):
    train_lines.append(f"{i}\n")
    list_lines.append(f"{i}\n")

# 写入新生成的文件
with open(os.path.join(output_dir, "train.txt"), 'w') as f:
    f.writelines(train_lines)

with open(os.path.join(output_dir, "list.txt"), 'w') as f:
    f.writelines(list_lines)
