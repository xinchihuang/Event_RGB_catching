# Event RGB Catching

事件相机与 RGB 相机的球体跟踪、轨迹拟合和接球系统，支持原始数据采集、离线回放及实时控制。

## Demo video

<!-- 发布前替换为公开视频链接或 GitHub 可访问的视频资源；检查画面与音频中的个人信息。 -->
**演示视频：待补充。**

## 快速开始

在项目根目录运行。新机器先完成 [Ubuntu 安装指南](UBUNTU22_INSTALL.md)，再配置数据路径：

```bash
cp -n config/paths.local.example.yaml config/paths.local.yaml
# 编辑 paths.local.yaml 中的 data_root，指向本机数据根目录。
./run_record.sh --task normal --test_mode Event --record_index 0 --no-gui
```

`paths.local.yaml` 不提交 Git。也可用 `EVENT_RGB_DATA_ROOT=/path/to/data` 临时覆盖。
回放需要已预处理的 `processed_data/saved_data_l/{event,rgb}`；缺少目录会明确报错，不自动寻找其他数据。

`run_record.sh` 默认使用 PATH 中的 `python3`，可用 `--python /path/to/python` 或 `PYTHON_BIN` 指定环境。
[requirement.txt](requirement.txt) 是依赖清单，不是版本锁；Metavision、PySpin 需单独安装，Model 模式需要 CUDA、AMTTrack 和匹配的 spconv。

## 运行入口

| 入口 | 用途 | 默认 mode / task / index |
| --- | --- | --- |
| `run_record.sh` | 单组文件回放，运行完退出 | Event / normal / 0 |
| `record.py` | 回放底层入口，支持精确输入/输出路径 | Model / dark / 0 |
| `tracking.py` | 相机跟踪与落点预测，无电机 | Model / dark_clutter / 0 |
| `live_real.py` | 相机跟踪、预测并控制电机 | Model / dark_clutter / 0 |
| `record_compare.py` | 三种方法的轨迹导出与叠加回放 | Playback / clutter / 9 |
| `start_evaluation.sh` | 固定组合批量评估，默认无 GUI | 见下文 |
| `run_record.py` | 旧 JSON 配置驱动的批量回放 | `batch_config.json` |

场景：`normal=l`、`clutter=c`、`dark_clutter=dc`、`dark=d`。
方法：`Model` 使用模型 ROI 与事件跟踪，`Event` 使用事件跟踪，`RGB` 使用 RGB 跟踪；单次回放还支持 `Show`。

```bash
# 单组 Model 回放；删除 --no-gui 可显示窗口。
./run_record.sh --task dark --test_mode Model --record_index 0 --no-gui
# 仅相机跟踪；t 开始、w 等待、e 退出。
python3 tracking.py --task normal --test_mode Event
# 实时接球：此命令会启动相机及电机。
python3 live_real.py --task dark_clutter --test_mode Model
# 依次导出 Model/RGB/Event 轨迹，再叠加回放。
python3 record_compare.py --task normal --record_index 0 --test_mode Save --output-root evaluation_tools/comparison_run
python3 record_compare.py --task normal --record_index 0 --test_mode Playback --output-root evaluation_tools/comparison_run
```

`ENABLE_GUI=0` 关闭 Python 入口的 GUI；`run_record.sh` 使用 `--no-gui`。
对比入口 `All` 等于依次执行 Save 和 Playback，`--playback_speed` 控制播放速度。
回放 `--ratio` 是每帧 RGB 对应的事件包数，默认 `10`、必须为正整数；它不控制采集或播放速度。

## 批量评估

```bash
bash start_evaluation.sh --dry-run       # 只检查配置与数据覆盖
bash start_evaluation.sh --limit 1       # 每个组合跑第 0 组，需要 CUDA
bash start_evaluation.sh                 # 完整批量，无 GUI
./run_record.sh --batch                  # 同一批量，有 GUI
python3 run_record.py --batch_config batch_config.json  # 自定义 JSON 组合
```

新批量固定为 Model 四场景、RGB normal/clutter、Event normal/dark：每组合 0–99，共 800 次。
默认每组超时 180 秒；缺失数据跳过，失败/超时记录后继续，Ctrl+C 停止当前批次及其子进程。
`--limit N` 限制每组合前 N 个编号；`--timeout`、`--ratio`、`--data-root` 可覆盖相应参数。

新旧批量**每次创建独立目录**，默认在 `evaluation_tools/batch_runs/`，所有结果进入该批 `results/`。
`--output-root` 必须指向尚不存在的目录。新批量另存 `configs/`、`logs/`、`summary.csv`、`status.json`；
`latest.json` 指向最新批次。退出码：0 完成、2 有进程/输出错误、130 中断。
旧 JSON 批量保存 `batch_config.json`；可逐项设置 `ratio`、`record_dir`，CLI `--ratio` 优先。
同一批内重复方法/场景/编号仍使用同名结果文件。

绘图与历史评估见 [工具文档](tools/README.md#evaluation)。

## 数据与结果

两条数据流程独立，详见 [采集和数据工具](dataset_tools/README.md)：

```text
原始采集 ──→ 旋转、时间裁剪 ──→ processed_data ──→ 文件回放
        └─→ 训练集生成（自行旋转、对齐） ──→ 合并与标注
```

新预处理输出 `.dat`，回放兼容历史 processed `.raw`；同编号两种文件并存时优先 `.dat`。
回放只做 RGB 对齐；不能用扩展名判断事件是否已旋转，也不能直接回放原始相机 RAW。

| 输出 | 内容与位置 |
| --- | --- |
| 单次 shell 回放 | `evaluation_tools/batch_runs/manual/<方法>_<场景>/`；重复编号可能覆盖，可用 `EVALUATION_RESULTS_DIR` 分开保存 |
| Python 回放/无电机跟踪 | 默认 `${data_root}/test_data/evaluation_results/<方法>_<场景>/` |
| `ball_pixel_3D_records_ball/<id>.csv` | 无表头像素观测 `(u, v, size, event_time_s)` |
| `ball_pixel_3D_records/<id>.csv` | 无表头预测落点 `(X, Y, Z)`，米，接球平面 X=0 |
| `--save_csv` | 仅控制 GUI 检测框 CSV，写到工作目录 `bbox_detections/`；无 GUI 时不生成，不影响上述轨迹和落点输出 |
| 对比回放 | 默认 `${data_root}/demo`，或 `--output-root` 指定的轨迹目录 |

`live_real.py` 用预测触发电机，不保存拟合 worker 的上述轨迹/落点 CSV。
全零预测不能单独证明未检测到球，也可能来自拟合或边界筛选。

在线/对比入口收到退出请求或 worker 故障后统一停止，先等待 5 秒，再终止/杀死未退出进程并清理共享内存；
正常退出为 0，故障或强制停止为 1。对比 worker 失败时不导出最终 CSV。采集保存使用独立的 60 秒等待规则。

## 文档与验证

| 文档 | 负责内容 |
| --- | --- |
| [配置说明](config/README.md) | 路径、优先级、参数、旋转和坐标定义 |
| [实时配置](config_live/README.md) | 如何选择独立实时调参配置 |
| [数据工具](dataset_tools/README.md) | 采集、预处理、训练集、计数 |
| [调参与评估工具](tools/README.md) | HSV、DBSCAN、绘图及标签映射 |
| [安装指南](UBUNTU22_INSTALL.md) | Ubuntu 部署记录、SDK 与依赖排错 |
| [修改台账](REVIEW_TRACKER.md) | P1/P2 状态、验证范围和完整历史 |
| [维护约定](AGENTS.md) | 修改后如何记录与验证 |
| [公开前检查](OPEN_SOURCE_REVIEW.md) | 敏感信息、跟踪文件和历史内容的发布建议 |

2026-10-11：P2 已完成，78 项测试通过；真实单样本“原始录制→DAT/MP4→Event 无界面回放”通过。
采集保存、在线故障退出和 GUI 仍有实机待验证项；上述验证不代表预测精度或全部场景通过。

## Citation

<!-- 发布后填写论文标题、作者、会议/期刊、年份、DOI/arXiv 和正式 BibTeX；不要提交虚构引用。 -->
**论文链接与 BibTeX：待补充。**
