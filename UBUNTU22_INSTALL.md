# Ubuntu 22.04 安装与兼容性记录

本文保留历史部署版本、SDK 安装顺序和排错细节。2026-10-11 已同步解释器选择与多机器路径用法；版本组合未在本次重新安装验证。

环境已安装时，直接看 [快速开始](README.md#快速开始) 和 [路径配置](config/README.md#paths)。
新机器按下方第 0–9 步安装，再执行软件验收；不要把历史版本表当作自动更新的依赖锁。

- 记录日期：2026-07-29
- 目标系统：Ubuntu 22.04.5 LTS，amd64/x86_64
- 目标用途：`record.py` 数据回放和 `live_real.py` 实机运行
- ROS：不需要

## 1. 已确认可行的版本组合

| 组件 | 推荐版本 | 说明 |
| --- | --- | --- |
| Ubuntu | 22.04.5 LTS amd64 | Metavision、Spinnaker 和 NVIDIA 均有对应支持 |
| Python | 系统 CPython 3.10.x | 所有相机 SDK 和 Python wheel 的共同交集 |
| Metavision | OpenEB/SDK 5.3.1 | Ubuntu 22.04 官方支持 Python 3.9/3.10 |
| Spinnaker SDK | 4.3.0.189 Ubuntu 22.04 x64 | 从 Teledyne 官方下载 |
| PySpin | 4.3.0.189 CPython 3.10 x86_64 | 必须与 Spinnaker SDK 完全同版 |
| NumPy | 1.26.4 | 不使用 NumPy 2 |
| PyTorch | 2.10.0+cu128 | 当前项目已经验证 |
| Torchvision | 0.25.0+cu128 | 与 PyTorch 2.10.0 配套 |
| NVIDIA 驱动 | 570.26 或更高 | CUDA 12.8 GA 的保守驱动要求 |
| 系统 CUDA Toolkit | 不安装 | pip Torch 和预编译 spconv 不需要 |
| spconv | spconv-cu124 2.3.8 | 有 CPython 3.10 Linux wheel |
| OpenCV | 4.10.0.84 | 当前项目已经验证 |
| Matplotlib | Ubuntu 22.04 系统包 | 避免和系统 `mpl_toolkits` 混装 |
| bytecode | 0.17.0 | 0.18.1 要求 Python 3.11+ |

不要在 Ubuntu 22.04 上复制当前机器的 `.venv312`，也不要为了本项目额外安装
Python 3.12。Ubuntu 22.04 上应重新创建 Python 3.10 虚拟环境。

## 2. 官方资料

- [Metavision 操作系统、Python 和相机支持矩阵](https://docs.prophesee.ai/stable/installation/index.html)
- [Metavision SDK Ubuntu 安装说明](https://docs.prophesee.ai/stable/installation/linux.html)
- [OpenEB Ubuntu 软件包安装说明](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html)
- [Metavision EventsIterator API](https://docs.prophesee.ai/stable/api/python/core/event_io.html)
- [Metavision 事件帧生成 API](https://docs.prophesee.ai/stable/guides/frames_generators.html)
- [Spinnaker SDK 官方下载](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/GetResourcesSupportDownloads/)
- [PySpin 官方安装说明](https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/)
- [PyTorch 官方历史版本安装命令](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.8 驱动版本要求](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [Ubuntu NVIDIA 驱动安装说明](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/)
- [spconv 官方说明](https://github.com/traveller59/spconv)
- [spconv-cu124 2.3.8 CPython 3.10 wheel](https://pypi.org/project/spconv-cu124/2.3.8/)

## 3. 严格安装顺序

不要跳过前面的验证步骤。相机还未连接时，先完成软件导入和已有数据回放测试。

### 第 0 步：检查目标机器

```bash
lsb_release -ds
uname -m
lscpu | grep -m1 -o avx2
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
```

必须满足：

1. 系统是 Ubuntu 22.04，架构为 `x86_64`。
2. CPU 支持 AVX2。
3. 有 NVIDIA GPU；Model 模式的项目代码直接调用 `.cuda()`，不能无修改地使用 CPU。
4. GPU 能使用 R570 或更高驱动。建议使用 Turing、Ampere、Ada 或更新架构。

如果 `nvidia-smi` 不存在，先继续第 1 步安装驱动。

### 第 1 步：更新系统并安装 NVIDIA 驱动

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers list
```

从 `ubuntu-drivers list` 的实际输出中选择 R570 或更高版本。例如目标机器确实列出
570 时：

```bash
sudo ubuntu-drivers install nvidia:570
sudo reboot
```

重启后验证：

```bash
nvidia-smi
```

如果机器只能使用 R560，可考虑改用 PyTorch CUDA 12.6；为了与当前已验证环境一致，
优先使用 R570+ 和 PyTorch cu128。

不要安装 CUDA Toolkit 或系统 cuDNN。这里需要的是 NVIDIA 驱动，Torch wheel 会安装
自身需要的 CUDA 用户态运行库。

### 第 2 步：安装系统基础依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  curl \
  ffmpeg \
  libcanberra-gtk-module \
  libevdev-dev \
  libgl1 \
  libglib2.0-0 \
  libusb-1.0-0 \
  libusb-1.0-0-dev \
  libxcb-xinerama0 \
  libxkbcommon-x11-0 \
  mesa-utils \
  python3.10-dev \
  python3.10-venv \
  python3-gi \
  python3-matplotlib \
  software-properties-common
```

可选的语音依赖：

```bash
sudo apt install -y espeak-ng
```

### 第 3 步：安装 Metavision/OpenEB

本项目运行路径使用的是开放模块：

- `metavision_core.event_io.EventsIterator`
- `metavision_sdk_core.PeriodicFrameGenerationAlgorithm`
- `metavision_sdk_core.ColorPalette`

因此，使用 Prophesee 官方 OpenEB 包通常已经足够。安装官方签名密钥和软件源：

```bash
curl -L \
  https://propheseeai.jfrog.io/artifactory/api/security/keypair/prophesee-gpg/public \
  -o /tmp/propheseeai.jfrog.op.asc

sudo cp /tmp/propheseeai.jfrog.op.asc /etc/apt/trusted.gpg.d/
sudo add-apt-repository \
  'https://propheseeai.jfrog.io/artifactory/openeb-debian/'
sudo apt update
```

先检查仓库提供的版本：

```bash
apt-cache madison metavision-openeb
apt-cache policy metavision-openeb
```

目标是 5.3.1。如果仓库的完整版本字符串包含 Ubuntu 打包后缀，可以使用上一步显示的
完整字符串进行固定：

```bash
sudo apt install "metavision-openeb=<apt-cache显示的5.3.1完整版本字符串>"
```

如果候选版本正是 5.3.1，也可以直接安装：

```bash
sudo apt install -y metavision-openeb
```

安装后记录实际版本：

```bash
dpkg-query -W 'metavision-*' 'prophesee-*' 2>/dev/null
```

如果使用完整 Metavision SDK，需要先取得 Prophesee JFrog 账号和令牌，并按照官方
SDK 文档配置认证仓库，然后安装：

```bash
sudo apt install -y metavision-sdk
```

不要混用 Ubuntu 24.04 的 Metavision deb 或 Python 3.12 二进制。

Ubuntu 22.04 的 HDF5 插件路径为：

```bash
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH:+${HDF5_PLUGIN_PATH}:}/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins"
```

### 第 4 步：安装 FLIR Spinnaker SDK

从 Teledyne 官方下载页面取得：

1. `Spinnaker SDK 4.3.0.189 for Linux Ubuntu 22.04 x64`
2. `PySpin 4.3.0.189 for CPython 3.10 Linux x86_64`

下载需要 Teledyne 账号。不要从非官方 PyPI 包代替 PySpin。

解压 Spinnaker SDK 压缩包，进入解压目录，运行包内脚本：

```bash
sudo ./install_spinnaker.sh
```

这一阶段只安装系统 SDK。PySpin wheel 等虚拟环境和 NumPy 建好后再安装。

确认动态库：

```bash
ldconfig -p | grep -i spinnaker
```

如果安装器把 SDK 安装到 `/opt/spinnaker`，后续启动环境可能需要：

```bash
export LD_LIBRARY_PATH="/opt/spinnaker/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GENICAM_GENTL64_PATH="/opt/spinnaker/lib/spinnaker-gentl${GENICAM_GENTL64_PATH:+:${GENICAM_GENTL64_PATH}}"
```

实际路径应以安装包和 `ldconfig` 输出为准。

### 第 5 步：创建 Python 3.10 虚拟环境

Metavision deb 把 Python 模块装在 Ubuntu 系统目录，因此按照官方建议使用
`--system-site-packages`：

```bash
cd /path/to/Event_RGB_catching
python3.10 -m venv .venv310 --system-site-packages
export PYTHONNOUSERSITE=true

.venv310/bin/python -m pip install --upgrade pip wheel
```

重要规则：

- 始终设置 `PYTHONNOUSERSITE=true`，不要读取 `~/.local` 中的包。
- 不要设置指向其他 conda/venv 的 `PYTHONPATH`。
- 不要在这个虚拟环境中用 pip 安装或升级 Matplotlib。
- 使用 Ubuntu 的 `python3-matplotlib` 和它自带的 `mpl_toolkits`。

之前的 `Axis has no attribute '_set_ticklabels'` 错误就是系统 `mpl_toolkits` 与 pip
Matplotlib 混装产生的。

### 第 6 步：安装固定的 Python 运行依赖

```bash
export PYTHONNOUSERSITE=true

.venv310/bin/python -m pip install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  pandas==2.2.3 \
  scikit-learn==1.5.2 \
  opencv-python==4.10.0.84 \
  PyYAML==6.0.2 \
  h5py==3.12.1 \
  Pillow==12.3.0 \
  tensorboard==2.21.0 \
  tensorboardX==2.6.5 \
  easydict==1.13 \
  lmdb==2.3.0 \
  timm==0.6.13 \
  bytecode==0.17.0 \
  pynput==1.8.2 \
  pyttsx3==2.99 \
  python-can==4.6.1 \
  openpyxl
```

`pynput` 的 Linux `evdev` 依赖可能在本机编译，因此前面必须安装
`build-essential`、`python3.10-dev` 和 `libevdev-dev`。

不要执行仓库中的旧版 `AMTTrack/install.sh`。该脚本安装 Torch 1.12.1、
CUDA 11.3、`tb-nightly` 和旧 setuptools，与当前代码和新显卡环境不一致。

### 第 7 步：安装 PyTorch 和 spconv

安装当前已经验证的 Torch 组合：

```bash
.venv310/bin/python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装 spconv：

```bash
.venv310/bin/python -m pip install spconv-cu124==2.3.8
```

Torch 使用 cu128、spconv 使用 cu124 是允许的。spconv 2.x 不依赖 PyTorch 二进制
ABI，CUDA 11+ 的 minor 版本可以不同。Linux 预编译包不要求系统 CUDA Toolkit，
但要求足够新的 NVIDIA 驱动。

安装前后检查不能存在多个 spconv/cumm 变体：

```bash
.venv310/bin/python -m pip list | grep -E '^(spconv|cumm)'
```

预期只有：

- `spconv-cu124`
- 它自动安装的匹配 `cumm-cu124`

### 第 8 步：安装匹配的 PySpin wheel

确认当前解释器和 NumPy：

```bash
.venv310/bin/python -c \
  'import sys, numpy; print(sys.version); print(numpy.__version__)'
```

必须显示 Python 3.10 和 NumPy 1.26.4。

安装下载好的官方 wheel：

```bash
.venv310/bin/python -m pip install \
  /实际下载路径/spinnaker_python-4.3.0.189-cp310-cp310-linux_x86_64.whl
```

要求：

- wheel 标签必须是 `cp310-cp310-linux_x86_64`。
- wheel 版本必须是 4.3.0.189。
- 系统 Spinnaker SDK 也必须是 4.3.0.189。
- 不升级到 NumPy 2。

### 第 9 步：选择解释器与本机数据路径

无需修改启动脚本源码。`run_record.sh` 默认使用 PATH 中的 `python3`，支持
`--python` 或 `PYTHON_BIN`；批量入口使用 `PYTHON_BIN`。例如在项目根目录：

```bash
cp -n config/paths.local.example.yaml config/paths.local.yaml
# 编辑 paths.local.yaml，把 data_root 改为该机器的数据根目录。
./run_record.sh --python "$PWD/.venv310/bin/python" --test_mode Event --task normal --record_index 0 --no-gui
PYTHON_BIN="$PWD/.venv310/bin/python" bash start_evaluation.sh --dry-run
```

原始采集位于 `data_root/saved_data_raw/saved_data_l`，回放输入位于
`data_root/processed_data/saved_data_l`。也可用 `--data-dir` 指定完整的
processed 场景目录；不要把原始 RAW 数据直接送入回放。

`run_live_real.sh` 保留本机 `.venv312` 默认值，但支持环境覆盖：

```bash
VENV_DIR="$PWD/.venv310" bash run_live_real.sh
```

此命令会启动相机和电机，只在硬件准备好时运行。Spinnaker 默认查找所选
环境下的 `spinnaker-sdk/opt/spinnaker`；系统安装可指定
`SPINNAKER_RUNTIME_DIR=/opt/spinnaker`，或按实际安装设置
`LD_LIBRARY_PATH` / `GENICAM_GENTL64_PATH`。也可直接用选定解释器执行
`tracking.py`（相机跟踪，无电机）或 `live_real.py`（接球控制）。

## 4. 软件验收

### 4.1 核心导入和版本检查

```bash
cd /path/to/Event_RGB_catching
export PYTHONNOUSERSITE=true

.venv310/bin/python - <<'PY'
import sys
import numpy
import cv2
import torch
import torchvision
import PySpin

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from spconv.pytorch.utils import PointToVoxel

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == "1.26.4"
assert torch.__version__.startswith("2.10.0")
assert torchvision.__version__.startswith("0.25.0")
assert torch.cuda.is_available()

print("GPU:", torch.cuda.get_device_name(0))
print("All core imports passed")
PY
```

检查依赖元数据：

```bash
.venv310/bin/python -m pip check
```

### 4.2 CUDA 计算检查

```bash
.venv310/bin/python - <<'PY'
import torch

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA tensor test passed:", y.shape, y.device)
PY
```

### 4.3 无相机数据回放检查

确认本机 `data_root/processed_data/saved_data_l` 可读，并包含第 5 组已预处理数据，然后运行：

```bash
./run_record.sh --python "$PWD/.venv310/bin/python" \
  --record_index 5 \
  --test_mode Model \
  --task normal
```

预期：

1. 读取 `saved_data_l/rgb/5.mp4` 和对应事件数据。
2. 打开 `Event & RGB Tracking` 窗口。
3. 显示 RGB、Event、模型 ROI/检测框。
4. 模型子进程没有 Torch、spconv、TensorBoard 或 Matplotlib 导入错误。

### 4.4 相机连接后的 PySpin 检查

```bash
.venv310/bin/python - <<'PY'
import PySpin

system = PySpin.System.GetInstance()
cameras = system.GetCameras()
try:
    print("FLIR camera count:", cameras.GetSize())
finally:
    cameras.Clear()
    system.ReleaseInstance()
PY
```

### 4.5 事件相机连接后的检查

先用 Metavision Studio 或官方 viewer 确认相机能被 SDK 打开，再运行：

```bash
.venv310/bin/python - <<'PY'
from metavision_core.event_io import EventsIterator

iterator = EventsIterator(input_path="", delta_t=1000)
print("Event camera size:", iterator.get_size())
events = next(iter(iterator))
print("First event packet:", events.shape)
PY
```

最后启动：

```bash
./run_live_real.sh
```

## 5. 已知冲突和禁止组合

### 不使用 Python 3.12

Ubuntu 22.04 的 Metavision 预编译 Python 绑定面向 3.9/3.10，PySpin 也应使用官方
CPython 3.10 wheel。不要把 Ubuntu 24.04/Python 3.12 的二进制复制过去。

### 不使用 NumPy 2

PySpin 官方安装说明要求使用 `numpy<2`。固定 `numpy==1.26.4`，安装其他包后重新
检查版本，防止被依赖解析器升级。

### 不混装 Matplotlib

不要同时使用：

- `/usr/lib/python3/dist-packages/mpl_toolkits`
- 虚拟环境中的另一版本 `matplotlib`

Ubuntu 22.04 推荐使用系统 `python3-matplotlib`，并且不要在 venv 中执行
`pip install --upgrade matplotlib`。

### 不运行旧版 AMTTrack 安装脚本

旧脚本中的 Torch 1.12.1/CUDA 11.3 不是本次部署版本。当前代码的 checkpoint 加载还
显式使用了较新 PyTorch 的 `weights_only=False` 参数。

### 不安装多套 CUDA

`nvidia-smi` 右上角显示的 CUDA 数字代表驱动能够支持的最高 CUDA API，并不表示
已经安装了 `/usr/local/cuda`。本项目的 pip Torch 和 spconv wheel 不需要额外 CUDA
Toolkit。

### 不混装 Spinnaker/PySpin 版本

Spinnaker SDK 和 PySpin 必须完整版本一致。不要组合 4.3.0.189 SDK 与 4.4.x wheel，
也不要用 Ubuntu 24.04 包替代 Ubuntu 22.04 包。

### 相机型号仍需最终确认

软件栈可以确认兼容，但实际连接前还要知道：

- Prophesee 事件相机的准确 EVK/传感器型号。
- FLIR 相机的准确型号和 USB3/GigE 接口类型。
- NVIDIA GPU 的准确型号、显存和可用驱动分支。

如果是旧 EVK1/EVK2，可能必须根据 Metavision 相机支持矩阵固定旧 SDK；EVK3/EVK4
可以优先使用 5.3.1。

## 6. 建议保存的安装结果

所有测试通过后，在目标机器保存完整环境记录：

```bash
mkdir -p environment_records

lsb_release -a > environment_records/os.txt 2>&1
uname -a > environment_records/kernel.txt
nvidia-smi > environment_records/nvidia-smi.txt
dpkg-query -W 'metavision-*' 'prophesee-*' > \
  environment_records/metavision-packages.txt 2>&1
.venv310/bin/python -m pip freeze > \
  environment_records/python310-freeze.txt
```

同时单独保存从 Teledyne 下载的 Spinnaker SDK 和 PySpin 安装包文件名及其 SHA256：

```bash
sha256sum /安装包实际路径/*spinnaker* > \
  environment_records/spinnaker-sha256.txt
```

这样可在重装或迁移时复现相同环境。
