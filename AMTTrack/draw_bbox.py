#!/usr/bin/env python3
"""
draw_bbox.py  ——  可视化 STARKProcessing 后的 search patch 及其 GT 框
用来排查 “GT 坐标错位 / 归一化比例错” 等常见训练不收敛问题。

用法:
    python draw_bbox.py  experiments/amttrack/amttrack_ball.yaml
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))        # 项目根到 PYTHONPATH

import argparse, yaml, torch, cv2, numpy as np
from easydict import EasyDict as edict

# ─── AMTTrack libs ───────────────────────────────────────────────────────────
from lib.train.base_functions import build_dataloaders
from lib.train.admin.settings import Settings
from lib.utils.box_ops import box_cxcywh_to_xyxy

# ─── 反归一化工具 ────────────────────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize_img(t: torch.Tensor) -> np.ndarray:
    img = (t * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    img = (img * 255).byte().cpu().permute(1, 2, 0).numpy()
    return img[:, :, ::-1].copy()                            # RGB→BGR

# ─── YAML → AttrDict (递归) ─────────────────────────────────────────────────
def dict2edict(d):
    if not isinstance(d, dict):
        return d
    return edict({k: dict2edict(v) for k, v in d.items()})

# ─── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visual‑debug STARKProcessing output")
parser.add_argument("config", help="Path to YAML config (experiments/…*.yaml)")
parser.add_argument("-n", "--num", type=int, default=5, help="patches to draw")
args = parser.parse_args()

CFG_PATH   = Path(args.config).expanduser()
N_SAMPLES  = args.num

# ─── 读取 YAML 并递归转换 ───────────────────────────────────────────────────
with open(CFG_PATH, "r") as f:
    cfg = dict2edict(yaml.safe_load(f))

# ─── Settings 对象初始化 ────────────────────────────────────────────────────
settings = Settings()
settings.local_rank   = 0
settings.device       = "cuda" if torch.cuda.is_available() else "cpu"
settings.project_path = Path(__file__).resolve().parent

# ─── 构建 DataLoader (训练集) ───────────────────────────────────────────────
train_loader, _ = build_dataloaders(cfg, settings)

# ─── 可视化输出目录 ────────────────────────────────────────────────────────
save_dir = Path("debug_vis"); save_dir.mkdir(exist_ok=True)

# ─── 遍历批次并绘制 ─────────────────────────────────────────────────────────
drawn = 0
for batch in train_loader:
    imgs   = batch["search_images"]            # [B, 3, 320, 320]
    bboxes = batch["search_anno"]              # [B, 4] 归一化 cxcywh

    for i in range(imgs.size(0)):
        img_np = denormalize_img(imgs[i])
        h, w   = img_np.shape[:2]

        # 归一化 → 像素坐标
        xyxy = box_cxcywh_to_xyxy(bboxes[i].unsqueeze(0))[0]
        xyxy *= torch.tensor([w, h, w, h])
        x1, y1, x2, y2 = xyxy.round().int().tolist()

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(save_dir / f"patch_{drawn}.png"), img_np)

        drawn += 1
        if drawn >= N_SAMPLES:
            break
    if drawn >= N_SAMPLES:
        break

print(f"[✓] {drawn} patches written to {save_dir}/")
