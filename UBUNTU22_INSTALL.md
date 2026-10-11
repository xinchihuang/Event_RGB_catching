# Ubuntu 22.04 installation and compatibility record

This document preserves historical deployment versions, SDK installation order, and troubleshooting details. Interpreter selection and portable paths were updated on 2026-10-11; the version combination was not reinstalled or revalidated during that update.

For an existing environment, see [Quick start](README.md#quick-start) and [path configuration](config/README.md#paths).
On a new machine, follow steps 0–9, then run the software checks. The historical version table is not an automatically maintained dependency lockfile.

- Recorded: 2026-07-29
- Target: Ubuntu 22.04.5 LTS, amd64/x86_64
- Use cases: `record.py` replay and `live_real.py` hardware operation
- ROS: not required

## 1. Previously validated version combination

| Component | Recorded version | Notes |
| --- | --- | --- |
| Ubuntu | 22.04.5 LTS amd64 | Supported by the corresponding Metavision, Spinnaker, and NVIDIA packages |
| Python | System CPython 3.10.x | Common version supported by the camera SDKs and Python wheels |
| Metavision | OpenEB/SDK 5.3.1 | Ubuntu 22.04 packages support Python 3.9/3.10 |
| Spinnaker SDK | 4.3.0.189 Ubuntu 22.04 x64 | Download from Teledyne |
| PySpin | 4.3.0.189 CPython 3.10 x86_64 | Must exactly match the Spinnaker SDK version |
| NumPy | 1.26.4 | Do not use NumPy 2 |
| PyTorch | 2.10.0+cu128 | Previously validated for this project |
| Torchvision | 0.25.0+cu128 | Paired with PyTorch 2.10.0 |
| NVIDIA driver | 570.26 or later | Conservative requirement for CUDA 12.8 GA |
| System CUDA Toolkit | Not installed | Not required by pip Torch or prebuilt spconv |
| spconv | spconv-cu124 2.3.8 | CPython 3.10 Linux wheel available |
| OpenCV | 4.10.0.84 | Previously validated for this project |
| Matplotlib | Ubuntu 22.04 system package | Avoid mixing with a different system `mpl_toolkits` |
| bytecode | 0.17.0 | 0.18.1 requires Python 3.11+ |

Do not copy the current machine's `.venv312` to Ubuntu 22.04 or install Python 3.12 for this setup.
Create a new Python 3.10 virtual environment on Ubuntu 22.04.

## 2. Official references

- [Metavision OS, Python, and camera support matrix](https://docs.prophesee.ai/stable/installation/index.html)
- [Metavision SDK installation on Ubuntu](https://docs.prophesee.ai/stable/installation/linux.html)
- [OpenEB Ubuntu package installation](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html)
- [Metavision EventsIterator API](https://docs.prophesee.ai/stable/api/python/core/event_io.html)
- [Metavision event frame generation API](https://docs.prophesee.ai/stable/guides/frames_generators.html)
- [Spinnaker SDK downloads](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/GetResourcesSupportDownloads/)
- [PySpin installation guide](https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/)
- [PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.8 driver requirements](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [Ubuntu NVIDIA driver installation](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/)
- [spconv documentation](https://github.com/traveller59/spconv)
- [spconv-cu124 2.3.8 CPython 3.10 wheel](https://pypi.org/project/spconv-cu124/2.3.8/)

## 3. Installation order

Do not skip verification between steps. If cameras are not connected, first check software imports and replay existing data.

### Step 0: Inspect the target machine

```bash
lsb_release -ds
uname -m
lscpu | grep -m1 -o avx2
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
```

Requirements:

1. Ubuntu 22.04 on `x86_64`.
2. CPU with AVX2 support.
3. NVIDIA GPU. Model mode calls `.cuda()` directly and cannot run on CPU without code changes.
4. GPU supported by R570 or newer drivers. Turing, Ampere, Ada, or newer architectures are recommended for this recorded setup.

If `nvidia-smi` is unavailable, install the driver in step 1 first.

### Step 1: Update the system and install the NVIDIA driver

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers list
```

Choose R570 or later from the actual `ubuntu-drivers list` output. For example, if 570 is listed:

```bash
sudo ubuntu-drivers install nvidia:570
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

If the machine supports only R560, consider PyTorch CUDA 12.6. Prefer R570+ with PyTorch cu128 to match the previously validated environment.
Do not install CUDA Toolkit or system cuDNN for this setup: the NVIDIA driver is required, while the Torch wheel installs its CUDA user-space runtime dependencies.

### Step 2: Install system dependencies

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

Optional speech dependency:

```bash
sudo apt install -y espeak-ng
```

### Step 3: Install Metavision/OpenEB

The project's runtime uses these open modules:

- `metavision_core.event_io.EventsIterator`
- `metavision_sdk_core.PeriodicFrameGenerationAlgorithm`
- `metavision_sdk_core.ColorPalette`

Prophesee's official OpenEB packages are therefore generally sufficient. Install the signing key and repository:

```bash
curl -L \
  https://propheseeai.jfrog.io/artifactory/api/security/keypair/prophesee-gpg/public \
  -o /tmp/propheseeai.jfrog.op.asc

sudo cp /tmp/propheseeai.jfrog.op.asc /etc/apt/trusted.gpg.d/
sudo add-apt-repository \
  'https://propheseeai.jfrog.io/artifactory/openeb-debian/'
sudo apt update
```

Check available versions first:

```bash
apt-cache madison metavision-openeb
apt-cache policy metavision-openeb
```

The recorded target is 5.3.1. If the package includes an Ubuntu packaging suffix, pin the complete version string shown above:

```bash
sudo apt install "metavision-openeb=<full-5.3.1-version-from-apt-cache>"
```

If the candidate is already 5.3.1, install directly:

```bash
sudo apt install -y metavision-openeb
```

Record installed versions:

```bash
dpkg-query -W 'metavision-*' 'prophesee-*' 2>/dev/null
```

For the full Metavision SDK, obtain a Prophesee JFrog account/token and configure the authenticated repository according to the official SDK guide, then install:

```bash
sudo apt install -y metavision-sdk
```

Do not mix Ubuntu 24.04 Metavision packages or Python 3.12 binaries into this environment.
The Ubuntu 22.04 HDF5 plugin path is:

```bash
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH:+${HDF5_PLUGIN_PATH}:}/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins"
```

### Step 4: Install FLIR Spinnaker SDK

Download from Teledyne:

1. `Spinnaker SDK 4.3.0.189 for Linux Ubuntu 22.04 x64`
2. `PySpin 4.3.0.189 for CPython 3.10 Linux x86_64`

A Teledyne account is required. Do not substitute an unofficial PyPI package for PySpin.
Extract the Spinnaker SDK archive, enter its directory, and run the included installer:

```bash
sudo ./install_spinnaker.sh
```

Install only the system SDK at this stage. Install the PySpin wheel after creating the virtual environment and pinning NumPy.
Check the shared library:

```bash
ldconfig -p | grep -i spinnaker
```

For an installation under `/opt/spinnaker`, startup may require:

```bash
export LD_LIBRARY_PATH="/opt/spinnaker/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GENICAM_GENTL64_PATH="/opt/spinnaker/lib/spinnaker-gentl${GENICAM_GENTL64_PATH:+:${GENICAM_GENTL64_PATH}}"
```

Use the actual package location and `ldconfig` output to determine paths.

### Step 5: Create a Python 3.10 virtual environment

Metavision deb packages install Python modules into system directories. Use `--system-site-packages` as recommended by the SDK instructions:

```bash
cd /path/to/Event_RGB_catching
python3.10 -m venv .venv310 --system-site-packages
export PYTHONNOUSERSITE=true

.venv310/bin/python -m pip install --upgrade pip wheel
```

Environment rules:

- Set `PYTHONNOUSERSITE=true` to avoid packages from `~/.local`.
- Do not point `PYTHONPATH` at another conda/venv environment.
- Do not install or upgrade Matplotlib with pip in this environment.
- Use Ubuntu's `python3-matplotlib` and its matching `mpl_toolkits`.

The previously observed `Axis has no attribute '_set_ticklabels'` error resulted from mixing system `mpl_toolkits` with pip Matplotlib.

### Step 6: Install pinned Python dependencies

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

The Linux `evdev` dependency of `pynput` may compile locally, requiring `build-essential`, `python3.10-dev`, and `libevdev-dev` from step 2.
Do not run the legacy `AMTTrack/install.sh`: it installs Torch 1.12.1, CUDA 11.3, `tb-nightly`, and old setuptools, which do not match this deployment.

### Step 7: Install PyTorch and spconv

Install the previously validated Torch combination:

```bash
.venv310/bin/python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install spconv:

```bash
.venv310/bin/python -m pip install spconv-cu124==2.3.8
```

This recorded setup uses Torch cu128 with spconv cu124. spconv 2.x does not depend on the PyTorch binary ABI and allows different CUDA 11+ minor versions.
Linux prebuilt packages do not require a system CUDA Toolkit, but do require a sufficiently recent NVIDIA driver.
Check that multiple spconv/cumm variants are not installed:

```bash
.venv310/bin/python -m pip list | grep -E '^(spconv|cumm)'
```

Expected packages:

- `spconv-cu124`
- Its automatically installed matching `cumm-cu124`

### Step 8: Install the matching PySpin wheel

Check the interpreter and NumPy:

```bash
.venv310/bin/python -c \
  'import sys, numpy; print(sys.version); print(numpy.__version__)'
```

The output must show Python 3.10 and NumPy 1.26.4.
Install the downloaded official wheel:

```bash
.venv310/bin/python -m pip install \
  /path/to/downloads/spinnaker_python-4.3.0.189-cp310-cp310-linux_x86_64.whl
```

Requirements:

- Wheel tag: `cp310-cp310-linux_x86_64`.
- Wheel version: 4.3.0.189.
- System Spinnaker SDK version: also 4.3.0.189.
- Do not upgrade to NumPy 2.

### Step 9: Select the interpreter and local data paths

No startup script edits are needed. `run_record.sh` defaults to `python3` from PATH and accepts `--python` or `PYTHON_BIN`; the batch entrypoint uses `PYTHON_BIN`.
For example, from the project root:

```bash
cp -n config/paths.local.example.yaml config/paths.local.yaml
# Set data_root in paths.local.yaml to this machine's data directory.
./run_record.sh --python "$PWD/.venv310/bin/python" --test_mode Event --task normal --record_index 0 --no-gui
PYTHON_BIN="$PWD/.venv310/bin/python" bash start_evaluation.sh --dry-run
```

Raw capture is under `data_root/saved_data_raw/saved_data_l`; replay input is under `data_root/processed_data/saved_data_l`.
Use `--data-dir` to select a complete processed scene directory. Do not send raw camera RAW files directly to replay.

`run_live_real.sh` retains its local `.venv312` default but accepts an environment override:

```bash
VENV_DIR="$PWD/.venv310" bash run_live_real.sh
```

This starts cameras and motors; run it only when the hardware is ready. Spinnaker defaults to `spinnaker-sdk/opt/spinnaker` under the selected environment.
For a system installation, set `SPINNAKER_RUNTIME_DIR=/opt/spinnaker` or configure `LD_LIBRARY_PATH` / `GENICAM_GENTL64_PATH` for the actual installation.
You can also run `tracking.py` (camera tracking without motors) or `live_real.py` (catching control) directly with the selected interpreter.

## 4. Software validation

### 4.1 Core imports and versions

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

Check dependency metadata:

```bash
.venv310/bin/python -m pip check
```

### 4.2 CUDA computation

```bash
.venv310/bin/python - <<'PY'
import torch

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA tensor test passed:", y.shape, y.device)
PY
```

### 4.3 Replay without cameras

Ensure `data_root/processed_data/saved_data_l` is readable and contains preprocessed recording 5, then run:

```bash
./run_record.sh --python "$PWD/.venv310/bin/python" \
  --record_index 5 \
  --test_mode Model \
  --task normal
```

Expected behavior:

1. Reads `saved_data_l/rgb/5.mp4` and matching events.
2. Opens the `Event & RGB Tracking` window.
3. Displays RGB, Event, and model ROI/detection boxes.
4. Model workers have no Torch, spconv, TensorBoard, or Matplotlib import errors.

### 4.4 PySpin with a connected camera

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

### 4.5 Connected event camera

First confirm that Metavision Studio or an official viewer can open the camera, then run:

```bash
.venv310/bin/python - <<'PY'
from metavision_core.event_io import EventsIterator

iterator = EventsIterator(input_path="", delta_t=1000)
print("Event camera size:", iterator.get_size())
events = next(iter(iterator))
print("First event packet:", events.shape)
PY
```

Finally start live operation with the selected environment:

```bash
VENV_DIR="$PWD/.venv310" ./run_live_real.sh
```

## 5. Known conflicts and unsupported combinations

### Python 3.12

Ubuntu 22.04 Metavision prebuilt Python bindings target 3.9/3.10; use the official CPython 3.10 PySpin wheel.
Do not copy Ubuntu 24.04/Python 3.12 binaries into this setup.

### NumPy 2

The PySpin installation instructions for this setup require `numpy<2`. Pin `numpy==1.26.4` and recheck after installing other packages so dependency resolution does not silently upgrade it.

### Mixed Matplotlib installations

Do not combine:

- `/usr/lib/python3/dist-packages/mpl_toolkits`
- A different `matplotlib` version in the virtual environment

For this Ubuntu 22.04 setup, use system `python3-matplotlib` and do not run `pip install --upgrade matplotlib` in the venv.

### Legacy AMTTrack installer

The old script's Torch 1.12.1/CUDA 11.3 combination is not this deployment's version set. Current checkpoint loading explicitly passes `weights_only=False` when the installed PyTorch supports that parameter.

### Multiple CUDA installations

The CUDA version displayed by `nvidia-smi` is the highest CUDA API supported by the driver; it does not mean `/usr/local/cuda` is installed.
The project's pip Torch and spconv wheels do not need an additional CUDA Toolkit.

### Mismatched Spinnaker/PySpin versions

Spinnaker SDK and PySpin must match exactly. Do not combine SDK 4.3.0.189 with a 4.4.x wheel or substitute Ubuntu 24.04 packages for Ubuntu 22.04 packages.

### Camera models still require confirmation

Before connecting hardware, confirm:

- Exact Prophesee EVK/sensor model.
- Exact FLIR model and USB3/GigE interface.
- NVIDIA GPU model, memory, and available driver branch.

Older EVK1/EVK2 devices may require an older SDK pinned according to the camera support matrix; EVK3/EVK4 can use 5.3.1 as the initial candidate for this deployment.

## 6. Save installation records

After all checks pass, save a complete environment record on the target machine:

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

Also retain the Teledyne Spinnaker SDK/PySpin package names and SHA256 hashes separately:

```bash
sha256sum /path/to/installers/*spinnaker* > \
  environment_records/spinnaker-sha256.txt
```

These records help reproduce the environment after reinstallation or migration.
