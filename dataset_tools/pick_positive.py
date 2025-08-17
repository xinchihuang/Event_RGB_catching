import os
import shutil

root_dir = "/media/xinchi/data/Event_RGB_catching_data"
os.makedirs("/media/xinchi/data/Event_RGB_catching_data/dataset", exist_ok=True)
shutil.copy(f"{root_dir}/dataset_old/train.txt", f"{root_dir}/dataset/train.txt")
shutil.copy(f"{root_dir}/dataset_old/list.txt", f"{root_dir}/dataset/list.txt")  # 确认这行逻辑是否正确

for i in range(0, 100):
    data_path = os.path.join(root_dir, 'dataset_old', f"{i}")
    new_data_path = os.path.join(root_dir, 'dataset', f"{i}")

    absent_path = os.path.join(data_path, "absent.txt")
    ground_truth_path = os.path.join(data_path, "groundtruth.txt")
    new_absent_path = os.path.join(new_data_path, "absent.txt")
    new_ground_truth_path = os.path.join(new_data_path, "groundtruth.txt")

    os.makedirs(new_data_path, exist_ok=True)
    os.makedirs(os.path.join(new_data_path, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(new_data_path, "event"), exist_ok=True)
    os.makedirs(os.path.join(new_data_path, "combined"), exist_ok=True)
    os.makedirs(os.path.join(new_data_path, "voxel"), exist_ok=True)

    image_count = 0
    with open(absent_path, 'r') as absent_file, \
         open(ground_truth_path, 'r') as ground_truth_file, \
         open(new_absent_path, 'w') as new_absent_file, \
         open(new_ground_truth_path, 'w') as new_ground_truth_file:

        for line_count, (absent_line, gt_line) in enumerate(zip(absent_file, ground_truth_file), start=1):
            if absent_line.strip() == '0':
                continue

            new_absent_file.write(absent_line)
            new_ground_truth_file.write(gt_line)

            shutil.copy(f"{data_path}/rgb/{line_count}.jpg", f"{new_data_path}/rgb/{image_count}.jpg")
            shutil.copy(f"{data_path}/event/{line_count}.jpg", f"{new_data_path}/event/{image_count}.jpg")
            shutil.copy(f"{data_path}/combined/{line_count}.jpg", f"{new_data_path}/combined/{image_count}.jpg")
            shutil.copy(f"{data_path}/voxel/{line_count}.mat", f"{new_data_path}/voxel/{image_count}.mat")

            image_count += 1
