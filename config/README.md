# Configuration

This page documents the shared interface for `config/` and `config_live/`. See the [project README](../README.md) for commands. Restart the program after changing parameters.

<a id="entrypoint-selection-and-overrides"></a>
## Configuration selection and precedence

All four main entrypoints (`record.py`, `record_compare.py`, `tracking.py`, `live_real.py`) use `config/` by default.
They recursively merge `camera_config.yaml → tracking_config.yaml → overlays/<scene_config>.yaml`, with later values overriding earlier ones.

| File | Settings |
| --- | --- |
| `camera_config.yaml` | Dimensions, calibration, alignment, and reading parameters |
| `tracking_config.yaml` | Shared tracking and trajectory fitting parameters |
| `overlays/*.yaml` | Scene/method overrides for ROI, HSV, templates, etc. |
| `paths.local.yaml` | Machine-specific data directories; excluded from Git |
| `legacy_full/*.yaml` | Historical full configurations; not selected automatically |

`ACTIVE_CONFIG_FILE` takes precedence over `--task/--test_mode` when selecting the exact overlay, with no fallback to another file of the same name:

```bash
ACTIVE_CONFIG_FILE=config_live/overlays/config_l.yaml python3 tracking.py --task normal --test_mode Event
```

- Selecting a file under `config_live/` also selects both base files from that directory. External overlays use the imported loader's default base files.
- CLI task/mode still determine the runtime method and output labels; keep them consistent with an explicit overlay.
- Without an explicit overlay, the entrypoint's task/mode select it. Python uses argparse defaults; shell entrypoints may first read defaults from `TASK/TEST_MODE`.
- `ACTIVE_TASK` is used for lazy configuration loading; it does not override a Python main entrypoint's argparse task default.
- All three YAML files must exist and contain mappings; use `{}` for an empty overlay. Errors are reported before resource creation; a failed load retains the last valid configuration.
- Both `Config` classes share accessors and return keys but maintain independent configuration state.
- The new batch runner saves merged snapshots and overrides input/output paths and `replay_apply_rotation=False`. The legacy JSON runner does not rewrite that rotation field. An explicit overlay applies to every task in the new batch.

<a id="paths"></a>
## Paths across machines

Create `config/paths.local.yaml` from the [example](paths.local.example.yaml):

```yaml
data_root: /path/to/event_rgb_data
```

`EVENT_RGB_DATA_ROOT` overrides the local `data_root`. `project_paths.py` expands `${data_root}` in configuration values.
Python path resolution supports `~` and resolves relative paths against the project root. Missing inputs raise errors; no legacy directory search or data migration occurs.

| Default directory relative to data_root | Purpose |
| --- | --- |
| `saved_data_raw/saved_data_{l,c,dc,d}` | Raw capture; input for preprocessing and training generation |
| `processed_data/saved_data_{l,c,dc,d}` | Rotated events and RGB input for replay |
| `saved_data/dataset_{l,c,dc,d}` | Training datasets |
| `saved_data/selected_dataset` | Merged and reviewed training data |
| `test_data/evaluation_results` | Default Python evaluation output and existing labels |
| `demo` | Comparison replay trajectories |

Stage paths follow: explicit CLI → corresponding environment variable → local configuration → default directory.
`Config.get_paths()` returns absolute paths: record_dir may come from an overlay, with input environment variables taking precedence;
evaluation_root is selected from environment, local configuration, then overlay.

| Option / environment variable | Meaning |
| --- | --- |
| `--data-root` / `DATA_ROOT` | **processed_data root**, not the entire data root |
| `--data-dir` (shell), `--record-dir` (Python) / `RECORD_DIR`, `DATA_DIR` | Full replay input directory for one scene |
| `--evaluation-root` / `EVALUATION_RESULTS_DIR` | Python replay output; shell defaults to the project's manual directory |
| `--save-dir` / `CAPTURE_DIR` | Capture output; preprocessing's `--save-dir` selects processed output instead |
| `--output-dir` / `DATASET_DIR` | Training dataset generation output |
| `--dataset-dir` / `SELECTED_DATASET_DIR` | Selected training dataset; read/write usage depends on the tool |
| `--output-root` / `COMPARISON_RESULTS_DIR` | Comparison replay output |

For both batch runners, `--output-root` selects a new batch directory; its results directory overrides inherited evaluation output environment variables.
If both exact-input environment variables are set, Python prefers `RECORD_DIR`, whereas shell prefers `DATA_DIR`; setting only one is recommended.
`run_record.sh --data-root` overrides inherited exact directories unless `--data-dir` is also supplied explicitly.
The legacy `saved_data dc` directory must be selected explicitly. Default paths depend on data_root; full explicit paths can be used independently.

Model templates `paths.template_jpg/npy` are relative to `tracking/`. The default checkpoint is
`AMTTrack/output/checkpoints/train/amttrack/amttrack_ball/AMTTrack_ep0199.pth.tar`.
See [evaluation tools](../tools/README.md#evaluation) for historical source, trajectory, and Excel settings.

Independent AMTTrack training paths are resolved by `AMTTrack/lib/train/admin/local.py`, without this project's data_root.
The workspace defaults to the AMTTrack directory, data to its `data/` subdirectory, and FELT to the current user's `~/FELT_dataset`.
Override these with `AMTTRACK_WORKSPACE_DIR`, `AMTTRACK_DATA_ROOT`, and `AMTTRACK_FELT_DIR`;
use `AMTTRACK_TENSORBOARD_DIR` and `AMTTRACK_PRETRAINED_DIR` for TensorBoard and pretrained networks.
Relative values resolve against AMTTrack and support `~`. Do not put personal absolute paths in shared source files.

## Parameter reference

These are shared defaults and meanings; the selected overlay may override values.

| Parameter | Meaning |
| --- | --- |
| `image.rgb_width/height` | Rotated RGB width/height, default 1080/1440 pixels |
| `image.event_width/height` | Rotated event width/height, default 720/1280; raw sensor width/height is 1280/720 |
| `alignment.event_pixel_size/rgb_pixel_size` | Pixel pitch used for alignment, in μm |
| `alignment.offset_x/offset_y` | RGB alignment offsets, in pixels |
| `alignment.y_coordinate_offset` | y translation in the fixed mounting rotation, default 1279 |
| `data_reading.delta_t` | Event packet duration, default 2000 μs |
| `data_reading.n_events` | Event count for SDK mixed/n_events modes; does not limit packet size in the current delta_t mode |
| `data_reading.send_interval/video_period` | Publication intervals in seconds, default 0.002/0.02 |
| `image_processing.*` | Event accumulation time, frame generation rate, binarization threshold, brightness/contrast, display scale |
| `camera_intrinsics.fx/fy/cx/cy` | Camera intrinsics in pixels |
| `camera_intrinsics.distortion_coeffs` | `[k1,k2,p1,p2,k3]`; the accessor returns this as `dist_coeffs` |
| `camera_extrinsics.camera_rpy_to/back` | World/camera rotation angles in degrees |
| `tracking.dbscan_eps/min_samples` | Clustering neighborhood radius in pixels / minimum points, default 15/100 |
| `tracking.max_ball_buffer` | Maximum tracking observations, default 1000; should match shared-memory capacity |
| `rgb_param.hsv_lower/upper` | RGB HSV thresholds; kernel_size and morphology_iterations control mask cleanup |
| `roi_range`, `default_bbox`, `backup_bbox` | ROI expansion and fallback bounding boxes, in pixels |
| `trajectory.gravity` | Acceleration along world Z, default −9.81 m/s²; must be finite |
| `trajectory.inlier_threshold` | 3D RANSAC inlier distance, default 0.35 m; must be positive and finite |
| `trajectory.impact_bounds_x` | Legacy key actually checking world **Y**, default open interval (−1.5, 1.5) m |
| `trajectory.impact_bounds_y` | Legacy key actually checking world **Z**, default open interval (0, 2) m |
| `trajectory.save_delay_sec` | Monotonic-clock delay after the first ball observation before fitting, saving, or triggering motors; default 0.15 s |

The fitting worker reads `gravity/inlier_threshold` from merged YAML at startup and passes them through current `MonoRansacDrag` fitting and prediction.
Capture and comparison replay without impact fitting do not use them; older experimental fitters retain their own constants.
`timing.*` and `buffers.*` are compatibility fields, not consumed directly by current main entrypoints/SharedMemoryManager. Changing them does not change shutdown deadlines or memory capacity.
See [live configuration](../config_live/README.md) for motor fields.

## Rotation boundaries

- Live event input always applies `(x,y)→(y,y_coordinate_offset−x)`, then RGB alignment. `apply_rotation` is an unused compatibility key.
- Device RAW logs are independent of preview rotation. Offline preprocessing and training generation each rotate raw events.
- Replay requires already-rotated processed data and only aligns it to RGB. Direct replay rejects `replay_apply_rotation=True`; the new batch runner explicitly sets it to False.
- Neither the extension nor legacy DAT headers establish orientation. New DAT files store rotated dimensions; legacy processed `.raw` remains readable.

<a id="observation-and-prediction-fields"></a>
## Observations and coordinates

Despite its name, `ball_pixel_3D_records_ball` stores `(u_px,v_px,size_px,event_time_s)` per row, without a header.
Event uses the smaller cluster side length as size; RGB uses the bounding-box size. Time comes from the current packet's last event timestamp, converted to seconds.
The current fitter treats size as apparent ball diameter and uses a physical diameter of 0.104 m, camera intrinsics, and the rotation matrix to recover world coordinates.

`ball_pixel_3D_records` stores world `(X,Y,Z)` in meters. The catching plane is X=0, Y is lateral, Z is vertical, and gravity acts along Z.
All-zero results can arise from missing observations, fit failure, or bounds filtering.
`save_delay_sec` starts when the fitting worker first sees a ball observation, independently of event timestamps and the start of tracking.
