# Event RGB Catching

Event + RGB capture, tracking, and landing prediction for ball catching.
This repo supports live operation with cameras and offline playback from recorded data.

## Installation

### Hardware requirements
- Event camera: Prophesee EVK4 HD, Metavision SDK 4.6.2.
- RGB camera: FLIR Blackfly S 16S2C, Spinnaker SDK (PySpin).
- Optional: PCAN USB + motor driver for live catching.
- Optional: CUDA-capable GPU for Model mode (AMTTrack + spconv).

### Python dependencies
Install the Python packages from `requirement.txt`:

```bash
pip install -r requirement.txt
```

For the verified Ubuntu 22.04/Python 3.10 version matrix, strict installation
order, camera SDK setup, and validation commands, see
[UBUNTU22_INSTALL.md](UBUNTU22_INSTALL.md).

## Configuration

The active configuration is merged as:
`camera_config + tracking_config + overlay`.

Default stack:
- `config/camera_config.yaml`
- `config/tracking_config.yaml`
- `config/overlays/*.yaml`

Live tuning stack (same schema):
- `config_live/camera_config.yaml`
- `config_live/tracking_config.yaml`
- `config_live/overlays/*.yaml`

For parameter meanings and tuning workflow, see:
- `config/README.md`
- `config_live/README.md`
- `tools/README.md`

## Run modes

### 1) Live catching from cameras (event + RGB + motor)
Reads from the event camera + FLIR camera and drives the motor.

```bash
python live_real.py --test_mode Model --task dark_clutter
```

### 2) Playback from recorded data (offline test)
Reads event + RGB from disk and runs the pipeline.

```bash
python record.py --record_index 0 --test_mode Model --task dark
```

Expected data layout under `record_dir` (set in the overlay):
- `event/{index}.raw` or `event/{index}.dat`
- `rgb/{index}.mp4`

### 3) Live tracking only (no motor)
Reads from cameras and runs tracking + landing prediction, without motor control.

```bash
python tracking.py --test_mode Event --task normal
```

Notes:
- Set `ENABLE_GUI=0` to disable the GUI.
- Keyboard control: `t` = start tracking, `w` = wait, `e` = exit.

## Test modes and tasks

`--test_mode` selects which pipeline is active:
- `Model`: AMTTrack ROI + event tracking.
- `Event`: event-only tracking.
- `RGB`: RGB ROI + RGB tracking.

`--task` selects the overlay set:
- `normal`, `dark`, `clutter`, `dark_clutter`

To change parameters, edit the overlay YAML that matches your task in
`config/overlays/` (or `config_live/overlays/` for the live stack).
You can also force a specific overlay with `ACTIVE_CONFIG_FILE`.

## Calibration

Calibration details live in `config/README.md`, `config_live/README.md`,
and `tools/README.md`.

## Evaluation tools

The `evaluation_tools/` folder contains scripts for comparing predictions,
plotting landing errors, and reviewing saved results. Use these after running
`record.py` or `live_real.py` to analyze performance and generate plots.

## Notes
- `config/` is the default stack used by the main scripts.
- `config_live/` mirrors the same schema and is available for live tuning.
