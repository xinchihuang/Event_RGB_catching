# Configuration (config_live/)

This stack mirrors `config/` but is kept separate for live tuning. The loader
merges:

1) `camera_config.yaml`
2) `tracking_config.yaml`
3) One overlay from `config_live/overlays/`

## Files
- `camera_config.yaml`: shared camera, alignment, and data-read defaults.
- `tracking_config.yaml`: shared tracking and trajectory defaults.
- `overlays/*.yaml`: per-task overrides (paths, ROI, HSV, etc.).
- `config_loader.py`: loader utility.

## Loading order
Overlay values always override base values when keys overlap.
You can select an overlay via `ACTIVE_CONFIG_FILE` or `--task`.

## Parameter reference

### camera_config.yaml

**image**
- `image.rgb_width`, `image.rgb_height`: RGB frame size in pixels.
- `image.event_width`, `image.event_height`: event sensor size in pixels.

**timing**
- `startup_delay`: initial delay before tracking starts (seconds).
- `reset_delay`: delay between resets (seconds).
- `shutdown_timeout`: grace period before forced shutdown (seconds).

**image_processing**
- `accumulation_time_us`: event frame accumulation window (microseconds).
- `fps`: target FPS for event frame generation (0.0 = use accumulation only).
- `binary_threshold`: threshold used for event frame binarization.
- `binary_max_value`: max value for binary threshold.
- `contrast_alpha`, `contrast_beta`: linear contrast/brightness for display.
- `brightness_multiplier`: optional RGB brightness multiplier (may be overridden in overlays).
- `display_scale`: UI display scale factor.

**clustering**
- `eps`, `min_samples`: legacy DBSCAN defaults used by some tools.

**buffers**
- `max_events`: shared-memory capacity for event packets.
- `ball_observations`: max stored observations for fitting.
- `ball_pixel_3d`: max stored 3D points.

**paths**
- `paths.evaluation_root`: output base for evaluation artifacts.

**alignment**
- `event_pixel_size`, `rgb_pixel_size`: physical pixel size (um) for alignment.
- `offset_x`, `offset_y`: pixel offsets from event to RGB coordinates.
- `y_coordinate_offset`: y-axis shift after rotation.

**data_reading**
- `delta_t`: event slice duration in microseconds.
- `n_events`: max number of events per slice.
- `send_interval`: publish interval for event stream (seconds).
- `video_period`: RGB publish interval (seconds).

**motor**
- `bitrate`, `motor_id`, `main_can_id`: CAN settings.
- `kp`, `kd`: motor control gains.
- `shaft_length`, `joint_height`: arm geometry for angle calculation.
- `movement_delay`: delay after move commands (seconds).

**camera_intrinsics**
- `fx`, `fy`, `cx`, `cy`: intrinsics used for 3D reconstruction.
- `distortion_coeffs`: lens distortion parameters.

**camera_extrinsics**
- `camera_rpy_to`, `camera_rpy_back`: rotation angles (degrees) for world transforms.

### tracking_config.yaml

**tracking**
- `dbscan_eps`: DBSCAN radius in pixels for event clustering.
- `dbscan_min_samples`: minimum points to form a cluster.
- `max_ball_buffer`: max stored ball points for fitting.

**trajectory**
- `inlier_threshold`: RANSAC inlier threshold.
- `gravity`: gravity constant used in fitting.
- `impact_bounds_x`, `impact_bounds_y`: valid impact bounds (meters).
- `save_delay_sec`: wait time before saving a record after tracking starts.

### overlays (config_live/overlays/*.yaml)

**paths**
- `paths.record_dir`: root folder for recorded data.
- `paths.template_npy`, `paths.template_jpg`: template files used by Model mode.

**rgb_param**
- `hsv_lower`, `hsv_upper`: HSV thresholds for RGB ROI detection.
- `kernel_size`: morphology kernel size for ROI mask cleanup.
- `morphology_iterations`: number of morphology iterations.

**roi_range**
- ROI expansion radius in pixels (used when expanding the detected ROI).

**default_bbox** / **backup_bbox**
- Fallback bounding boxes when detection is missing or unstable.

## Notes
- Update overlays for per-task tuning; keep shared values in base files.
- Restart the application to apply config changes.
