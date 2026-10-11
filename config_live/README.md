# Live tuning configuration

`config_live/` keeps independent live tuning values while sharing the [configuration interface, parameters, and coordinates](../config/README.md).
Even `tracking.py` and `live_real.py` default to `config/`; select this directory explicitly:

```bash
ACTIVE_CONFIG_FILE=config_live/overlays/config_l.yaml python3 tracking.py --task normal --test_mode Event
```

This loads the directory's `camera_config.yaml`, `tracking_config.yaml`, and selected overlay.
Scene suffixes are l/c/dc/d; overlays here are not split by Event/RGB/Model. Restart after changes, and keep CLI task/mode consistent with the selected scene and method.

| Settings | Location |
| --- | --- |
| Live calibration, dimensions, alignment, reading parameters | `camera_config.yaml` |
| Tracking and shared fitting parameters, including gravity/inlier_threshold | `tracking_config.yaml` |
| Scene-specific ROI, HSV, templates | `overlays/*.yaml` |
| Machine-specific data paths | Share `../config/paths.local.yaml`; do not create a second copy |

Motor fields are also retained: `bitrate/motor_id/main_can_id` are CAN settings, `kp/kd` are gains,
`shaft_length/joint_height` are mechanism dimensions in meters, and `movement_delay` is in seconds. Actual use depends on the motor call sites.
`timing.*` and `buffers.*` are compatibility fields not directly used by current main entrypoints or shared-memory allocation.

Live rotation follows the fixed camera mounting orientation; `apply_rotation` does not control it. Keep the [capture and replay boundaries](../config/README.md#rotation-boundaries) distinct.
