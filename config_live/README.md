# 实时调参配置

`config_live/` 用于独立保存实时调参值，共用 [配置接口、参数和坐标说明](../config/README.md)。
即使运行 `tracking.py` 或 `live_real.py`，默认仍选择 `config/`；使用本目录必须显式指定：

```bash
ACTIVE_CONFIG_FILE=config_live/overlays/config_l.yaml python3 tracking.py --task normal --test_mode Event
```

选择后加载本目录的 `camera_config.yaml`、`tracking_config.yaml` 和指定 overlay。
场景后缀为 l/c/dc/d，本目录不按 Event/RGB/Model 分出 overlay。修改后重启，CLI task/mode 应与选择的场景、运行方法匹配。

| 内容 | 放置位置 |
| --- | --- |
| 实时标定、尺寸、对齐和读取参数 | `camera_config.yaml` |
| 跟踪、gravity、inlier_threshold 等公共拟合参数 | `tracking_config.yaml` |
| 场景专用 ROI、HSV、模板等 | `overlays/*.yaml` |
| 本机数据路径 | 共用 `../config/paths.local.yaml`，不另建一份 |

本目录还保留 motor 参数：`bitrate/motor_id/main_can_id` 为 CAN 参数，`kp/kd` 为增益，
`shaft_length/joint_height` 为机构尺寸（米），`movement_delay` 为动作延时（秒）。实际使用以 motor 调用代码为准。
`timing.*` 与 `buffers.*` 是兼容字段，当前主入口和共享内存分配不直接使用。

实时旋转按固定安装方向执行，`apply_rotation` 不控制它；[采集与回放的边界](../config/README.md#旋转边界)保持独立。
