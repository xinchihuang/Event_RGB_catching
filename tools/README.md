# Tools

This folder contains lightweight utilities for tuning RGB HSV thresholds and
DBSCAN parameters used by the tracking pipeline. These scripts do not update
YAML files automatically; copy the suggested values into the appropriate
config overlay under `config/overlays/` or `config_live/overlays/`.

## hsv_selector.py

Interactive HSV sampler for RGB images or a live camera feed. Click a pixel to
print its HSV value and a suggested lower/upper range.

### Usage

Image file:

```bash
python tools/hsv_selector.py --source /path/to/frame.jpg
```

Camera index:

```bash
python tools/hsv_selector.py --source 0
```

Single pixel without GUI (image mode only):

```bash
python tools/hsv_selector.py --source /path/to/frame.jpg --x 320 --y 180
```

### Parameters
- `--source`: Image path or camera index (default: `l.jpg`).
- `--x`, `--y`: Optional pixel coordinates for a single sample.
- `--dh`, `--ds`, `--dv`: Range deltas used to compute the suggested HSV bounds.

### How to apply
Use the printed `lower` and `upper` values to update:
- `rgb_param.hsv_lower`
- `rgb_param.hsv_upper`

These live in the per-task overlays (`config/overlays/*.yaml` or
`config_live/overlays/*.yaml`).

## interactive_dbscan_tuner.py

Interactive DBSCAN tuner for event data stored in a `.raw` file. It loads
alignment and tracking defaults from the active config overlay, clusters events,
and visualizes the result in a live window.

### Usage

```bash
python tools/interactive_dbscan_tuner.py /path/to/event.raw --task clutter
```

Optional display scale:

```bash
python tools/interactive_dbscan_tuner.py /path/to/event.raw --scale 0.8
```

### Keyboard controls
- `SPACE`: Play/pause auto-play
- `j` / `l`: Previous / next frame
- `u` / `o`: Fast backward / forward (-/+10 frames)
- `h` / `E`: First / last frame
- `e` / `q`: `eps` +1 / -1
- `d` / `a`: `min_samples` +5 / -5
- `r`: Reset parameters to config defaults
- `p`: Print current parameters
- `+` / `-`: Increase / decrease playback FPS
- `ESC`: Exit

### Parameter meaning
- `tracking.dbscan_eps` (`eps`): Neighborhood radius (pixels) used by DBSCAN.
  Larger values merge more points into clusters.
- `tracking.dbscan_min_samples` (`min_samples`): Minimum number of points in the
  `eps` neighborhood to form a cluster. Larger values reduce noise but may drop
  weak detections.
- `data_reading.delta_t`: Event slice duration in microseconds.
- `data_reading.n_events`: Max number of events per slice.
- `--scale`: UI display scale only (does not affect clustering).

### How to apply
When you find good values, update them in:
- `config/tracking_config.yaml` (default stack), or
- `config_live/tracking_config.yaml` (live stack).

If you need task-specific overrides, place them in the corresponding overlay.
