# 配置说明

本页统一说明 `config/` 与 `config_live/` 的公共接口。运行命令见 [项目 README](../README.md)，修改参数后重启程序。

<a id="entrypoint-selection-and-overrides"></a>
## 配置选择与优先级

四个主入口 `record.py`、`record_compare.py`、`tracking.py`、`live_real.py` 默认都使用 `config/`。
每次递归合并 `camera_config.yaml → tracking_config.yaml → overlays/<场景配置>.yaml`，后者覆盖前者。

| 文件 | 修改场景 |
| --- | --- |
| `camera_config.yaml` | 尺寸、标定、对齐、读取参数 |
| `tracking_config.yaml` | 跟踪与轨迹拟合公共参数 |
| `overlays/*.yaml` | 指定场景/方法的 ROI、HSV、模板等覆盖值 |
| `paths.local.yaml` | 每台机器的数据目录，不提交 Git |
| `legacy_full/*.yaml` | 保留的历史完整配置，默认入口不自动选用 |

`ACTIVE_CONFIG_FILE` 优先于 `--task/--test_mode` 选择准确的 overlay，不按同名文件回退：

```bash
ACTIVE_CONFIG_FILE=config_live/overlays/config_l.yaml python3 tracking.py --task normal --test_mode Event
```

- 选中 `config_live/` 内的文件时，两份 base 也来自该目录；外部 overlay 使用所导入 loader 的默认 base。
- CLI 的 task/mode 仍决定运行方法与输出标签，应与显式 overlay 匹配。
- 没有显式 overlay 时，由入口解析出的 task/mode 选择；Python 入口使用 argparse 默认值，shell 入口可先从 `TASK/TEST_MODE` 取默认值。
- `ACTIVE_TASK` 供延迟加载配置时使用，不覆盖 Python 主入口的 argparse 默认 task。
- 三份 YAML 都必须存在且为映射；空 overlay 写 `{}`。错误在创建资源前报告，加载失败保留上一份有效配置。
- 两套 `Config` 共用访问实现和返回键，分别保留自己的配置状态。
- 新批量保存合并后的快照，并覆盖输入/输出路径和 `replay_apply_rotation=False`；旧 JSON 批量不改写该旋转字段。显式 overlay 对新批量的全部任务生效。

<a id="paths"></a>
## 多机器路径

从 [示例](paths.local.example.yaml) 创建本机 `config/paths.local.yaml`：

```yaml
data_root: /path/to/event_rgb_data
```

`EVENT_RGB_DATA_ROOT` 优先于本机 `data_root`。配置中的 `${data_root}` 由 `project_paths.py` 展开；
Python 路径解析支持 `~`，相对路径以项目根目录为准。缺失输入报错，不自动寻找旧目录，也不迁移数据。

| 相对 data_root 的默认目录 | 用途 |
| --- | --- |
| `saved_data_raw/saved_data_{l,c,dc,d}` | 原始采集；预处理与训练生成的输入 |
| `processed_data/saved_data_{l,c,dc,d}` | 已旋转事件和 RGB 回放输入 |
| `saved_data/dataset_{l,c,dc,d}` | 训练集 |
| `saved_data/selected_dataset` | 合并、审核后的训练数据 |
| `test_data/evaluation_results` | Python 入口默认评估输出和原有标签 |
| `demo` | 对比回放轨迹 |

阶段路径按“显式 CLI → 对应环境变量 → 本机配置 → 默认目录”解析。
`Config.get_paths()` 返回绝对路径：record_dir 可来自 overlay，输入环境变量优先；
evaluation_root 按环境变量、本机配置、overlay 顺序选取。

| 选项 / 环境变量 | 含义 |
| --- | --- |
| `--data-root` / `DATA_ROOT` | **processed_data 根目录**，不是整个数据根目录 |
| `--data-dir`（shell）、`--record-dir`（Python） / `RECORD_DIR`、`DATA_DIR` | 某一个场景的完整回放目录 |
| `--evaluation-root` / `EVALUATION_RESULTS_DIR` | Python 回放结果目录；shell 默认使用项目内 manual 目录 |
| `--save-dir` / `CAPTURE_DIR` | 采集输出；预处理的 `--save-dir` 则指定预处理输出 |
| `--output-dir` / `DATASET_DIR` | 训练集生成输出 |
| `--dataset-dir` / `SELECTED_DATASET_DIR` | 默认选中训练集；按工具用途读写 |
| `--output-root` / `COMPARISON_RESULTS_DIR` | 对比回放输出 |

新旧批量 `--output-root` 指定本批新目录，批次 results 覆盖继承的评估输出环境变量。
Python 精确输入同时设置两环境变量时 `RECORD_DIR` 优先；shell 则优先 `DATA_DIR`，建议只设置一个。
`run_record.sh --data-root` 会覆盖继承的精确目录，除非同时显式给出 `--data-dir`。
旧 `saved_data dc` 目录只能显式指定。缺省路径依赖 data_root，完整显式路径可独立使用。

Model 模板 `paths.template_jpg/npy` 相对 `tracking/`；默认 checkpoint 位于
`AMTTrack/output/checkpoints/train/amttrack/amttrack_ball/AMTTrack_ep0199.pth.tar`。
历史源码、轨迹、Excel 配置见 [评估工具](../tools/README.md#evaluation)。

AMTTrack 独立训练路径由 `AMTTrack/lib/train/admin/local.py` 解析，不使用本项目的 data_root：
默认 workspace 为 AMTTrack 目录，数据在其 `data/`，FELT 在当前用户的 `~/FELT_dataset`。
可用 `AMTTRACK_WORKSPACE_DIR`、`AMTTRACK_DATA_ROOT`、`AMTTRACK_FELT_DIR` 覆盖；
TensorBoard 和预训练网络分别用 `AMTTRACK_TENSORBOARD_DIR`、`AMTTRACK_PRETRAINED_DIR`。
这些环境变量的相对路径以 AMTTrack 目录为基准，支持 `~`，不在共享源码里填写个人绝对路径。


## 参数速查

下表是公共默认含义；具体生效值还需看选中的 overlay。

| 参数 | 含义 |
| --- | --- |
| `image.rgb_width/height` | 旋转后 RGB 图像宽/高，默认 1080/1440 像素 |
| `image.event_width/height` | 旋转后事件图像宽/高，默认 720/1280；原始传感器宽/高为 1280/720 |
| `alignment.event_pixel_size/rgb_pixel_size` | 对齐公式中的像元尺寸，μm |
| `alignment.offset_x/offset_y` | RGB 对齐偏移，像素 |
| `alignment.y_coordinate_offset` | 固定安装旋转后的 y 平移，默认 1279 |
| `data_reading.delta_t` | 事件分包时间，默认 2000 μs |
| `data_reading.n_events` | SDK mixed/n_events 模式的事件数参数；当前 delta_t 模式不靠它限制包大小 |
| `data_reading.send_interval/video_period` | 发布间隔，秒，默认 0.002/0.02 |
| `image_processing.*` | 事件累计时间、生成帧率、二值化阈值、亮度/对比度、显示缩放 |
| `camera_intrinsics.fx/fy/cx/cy` | 相机内参，像素 |
| `camera_intrinsics.distortion_coeffs` | `[k1,k2,p1,p2,k3]`；访问接口返回键为 `dist_coeffs` |
| `camera_extrinsics.camera_rpy_to/back` | 世界/相机坐标变换的旋转角，度 |
| `tracking.dbscan_eps/min_samples` | 聚类邻域半径（像素）/最小点数，默认 15/100 |
| `tracking.max_ball_buffer` | 跟踪观测数上限，默认 1000；应匹配共享内存容量 |
| `rgb_param.hsv_lower/upper` | RGB HSV 阈值；kernel_size、morphology_iterations 控制掩膜清理 |
| `roi_range`、`default_bbox`、`backup_bbox` | ROI 扩展及备用检测框，像素 |
| `trajectory.gravity` | 世界 Z 轴加速度，默认 −9.81 m/s²，须为有限值 |
| `trajectory.inlier_threshold` | 三维 RANSAC 内点距离阈值，默认 0.35 m，须为正有限值 |
| `trajectory.impact_bounds_x` | 兼容旧名，实际检查世界 **Y**，默认开区间 (−1.5, 1.5) m |
| `trajectory.impact_bounds_y` | 兼容旧名，实际检查世界 **Z**，默认开区间 (0, 2) m |
| `trajectory.save_delay_sec` | 首次球观测后按单调时钟等待多久再拟合、保存或触发电机，默认 0.15 s |

`gravity/inlier_threshold` 在拟合 worker 启动时从合并 YAML 读取，贯穿当前 `MonoRansacDrag` 的拟合和预测。
采集与不做落点拟合的对比回放不使用它们；早期实验拟合器保留各自常量。
`timing.*`、`buffers.*` 是兼容字段，当前主入口/SharedMemoryManager 不直接消费，修改它们不会改变停止时限或内存容量。
实时专用 motor 字段见 [实时配置](../config_live/README.md)。

## 旋转边界

- 实时事件输入固定执行 `(x,y)→(y,y_coordinate_offset−x)`，再做 RGB 对齐；`apply_rotation` 是未使用的兼容键。
- 原始 RAW 日志不随预览旋转；离线预处理和训练生成各自旋转原始事件。
- 回放要求已经旋转的 processed 数据，只做 RGB 对齐。直接回放拒绝 `replay_apply_rotation=True`；新批量明确将其设为 False。
- 扩展名和旧 DAT 头部都不能证明方向。新 DAT 保存旋转后的尺寸，旧 processed `.raw` 保持可读。

<a id="observation-and-prediction-fields"></a>
## 观测与坐标

`ball_pixel_3D_records_ball` 虽有 3D 字样，实际每行是 `(u_px,v_px,size_px,event_time_s)`，没有表头。
Event 使用较小的聚类边长作为 size，RGB 使用检测框尺寸；时间取当前包最后一个事件时间戳并转为秒。
当前拟合器用 size 作为球的表观直径，物理球径 0.104 m，经内参和旋转矩阵恢复世界坐标。

`ball_pixel_3D_records` 存储世界 `(X,Y,Z)`，单位米；接球平面 X=0，Y 为横向，Z 为竖直，重力作用于 Z。
全零结果可能来自无观测、拟合失败或越界筛选。
`save_delay_sec` 从拟合 worker 首次看到球观测开始计时，与事件时间戳及开始跟踪的时刻不同。
