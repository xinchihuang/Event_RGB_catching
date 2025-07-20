from metavision_core.event_io import EventsIterator
import torch, os, scipy.io, numpy as np
from tqdm import tqdm
from spconv.pytorch.utils import PointToVoxel

def process_one_raw(raw_path, mat_dir,
                    save_voxel=5_000,
                    slice_us=33_333,        # ≈30 FPS
                    voxel_gen=None,
                    device=torch.device("cuda:0")):
    """把单个 .raw 文件切成若干 voxel 切片并存 .mat"""

    it = EventsIterator(raw_path, mode="delta_t",
                        delta_t=slice_us, start_ts=0)   # :contentReference[oaicite:1]{index=1}
    H, W = it.get_size()

    for idx, evs in enumerate(tqdm(it, desc=os.path.basename(raw_path))):
        if evs.size == 0:           # 空片段占位
            scipy.io.savemat(
                os.path.join(mat_dir, f"frame{idx:04d}.mat"),
                {"coor": np.zeros((100, 3)),
                 "features": np.zeros((100, 16))}
            )
            continue

        # 把 numpy 结构化数组 → Tensor，并完成极性映射
        t = torch.as_tensor(evs["t"], dtype=torch.float32, device=device).unsqueeze(1)

        x = torch.as_tensor(evs["x"].astype(np.int16), device=device).unsqueeze(1)
        y = torch.as_tensor(evs["y"].astype(np.int16), device=device).unsqueeze(1)

        p = torch.as_tensor(evs["p"].astype(np.int8), device=device).unsqueeze(1)
        p[p == 0] = -1  # 负极标为 -1

        # 时间归一化到 0–1000 us 避免溢出
        dt = t[-1] - t[0]
        t = ((t - t[0]) / dt * 1000) if dt > 0 else torch.zeros_like(t)

        points = torch.cat((t, x, y, p), dim=1)      # (N,4)
        coor, feats = transform_points_to_voxels(
            {'points': points},
            voxel_generator=voxel_gen,
            device=device,
            save_voxel=save_voxel                   # 传参而不是用全局
        )

        scipy.io.savemat(
            os.path.join(mat_dir, f"frame{idx:04d}.mat"),
            {"coor": coor.cpu().numpy(),
             "features": feats.cpu().numpy()}
        )

def transform_points_to_voxels(data_dict, voxel_generator, device,
                               save_voxel=5_000):
    """略微精简，并使 save_voxel 显式可调"""
    pts = data_dict['points'].to(device)
    shuffle = torch.randperm(pts.shape[0], device=device)
    pts = pts[shuffle]

    voxels, coords, num_pts = voxel_generator(pts)
    # 选密度最高的前 save_voxel 个 voxel
    if num_pts.shape[0] > save_voxel:
        _, idx = torch.topk(num_pts, save_voxel)
        voxels, coords, num_pts = voxels[idx], coords[idx], num_pts[idx]

    coords = coords.to(device)
    coords[:, [0, 1, 2]] = coords[:, [2, 1, 0]]  # (y,x,t)→(t,x,y)
    feats = voxels[:, :, 3]                      # 取极性通道

    return coords, feats

# -------- 入口 -------- #
if __name__ == "__main__":
    device = torch.device("cuda:0")
    raw_root   = "/home/xinchi/Event_RGB_catching_data/dataset"
    save_root  = "/home/xinchi/Event_RGB_catching_data/dataset"   # 与原目录平级
    os.makedirs(save_root, exist_ok=True)

    voxel_gen = PointToVoxel(
        vsize_xyz=[50, 10, 10],
        coors_range_xyz=[0, 0, 0, 1000, 345, 259],
        num_point_features=4,
        max_num_voxels=16_000,
        max_num_points_per_voxel=16,
        device=device)

    for folder in sorted(os.listdir(raw_root)):
        raw_file = os.path.join(raw_root, folder, f"event.raw")
        print(raw_file)
        if not os.path.isfile(raw_file):
            continue

        mat_dir = os.path.join(save_root, folder, f"{folder}_voxel")
        os.makedirs(mat_dir, exist_ok=True)
        process_one_raw(raw_file, mat_dir,
                        voxel_gen=voxel_gen,
                        device=device)
