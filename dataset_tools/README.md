# 数据采集与处理

先按 [路径配置](../config/README.md#paths) 设置 data_root。以下命令从项目根目录执行，`--task` 选择 normal/clutter/dark_clutter/dark。

## 两条独立流程

| 流程 | 输入 → 输出 | 事件变换 |
| --- | --- | --- |
| `collect_data.py` | 相机 → `saved_data_raw/saved_data_<场景>` | 设备 RAW 日志不旋转；事件预览与保存分开；RGB 旋转后保存 |
| `time_align_rewrite.py` | 原始目录 → `processed_data/saved_data_<场景>` | 旋转、时间裁剪；保存前不做 RGB 空间对齐 |
| 文件回放 | processed 目录 → 跟踪/落点结果 | 只做 RGB 对齐，不重复旋转 |
| `prepare_dataset.py` | 原始目录 → `saved_data/dataset_<场景>` | 自行旋转并对齐 RGB，生成训练帧/体素/标签 |

**prepare_dataset 不接在 time_align_rewrite 后面**，两者都读取原始采集。

## 采集与回放预处理

```bash
python3 dataset_tools/collect_data.py --task normal
python3 dataset_tools/time_align_rewrite.py --task normal --index 0 --count 1 --no-view
./run_record.sh --task normal --test_mode Event --record_index 0 --no-gui
```

采集按键：`r` 开始、`f` 完成保存、`q` 退出。事件与 RGB 共用固定编号，等待 `Saved pair` 后再开始下一次；
录制/保存时重复开始请求会被忽略。q、Ctrl+C、SIGTERM 都先请求保存，最多等待 60 秒；未完成配对、空视频或超时返回失败。

预处理新输出是 `event/<id>.dat` 与 `rgb/<id>.mp4`，事件坐标为 `(raw_y, raw_width−1−raw_x)`，
头部高×宽从 720×1280 变为 1280×720，保留时间戳和极性。预览的 RGB 空间对齐不写入事件文件。
历史 processed `.raw`（包含旧 DAT 内容/尺寸信息）仍可回放；同编号 `.dat` 优先，旧文件不删除。

| 路径选项 | 用途 |
| --- | --- |
| 采集 `--save-dir`（别名 `--output-dir`） | 指定原始保存目录 |
| 预处理 `--record-dir / --save-dir` | 指定原始输入与 processed 输出 |
| 训练生成 `--record-dir / --output-dir` | 指定原始输入与训练集输出 |

预处理默认 `index=0,count=20,ratio=10`，`--auto` 从起始编号处理到首个缺失配对；`--no-view` 关闭预览。
`--no-spatial-align` 只关闭预览对齐。预处理和训练生成拒绝相同或相互包含的输入/输出目录。

## 训练集与标注

```bash
python3 dataset_tools/prepare_dataset.py --task normal --index 0 --count 1
python3 dataset_tools/pick_positive.py
python3 dataset_tools/dataset_viewer.py
```

训练生成保留预览行为；不传编号时默认处理 13–99（index=13,count=87）。
`pick_positive.py` 默认合并四个场景到 `saved_data/selected_dataset`，可用重复的 `--source-dir` 选择输入、`--output-dir` 选择输出。

| 工具 | 用途 |
| --- | --- |
| `dataset_viewer.py` | 查看 RGB/Event/Combined/Voxel 并记录标记 |
| `update_absent_by_flags.py` | 按 frame_flags.txt 更新 absent.txt |
| `adjust_groundtruth_y.py` | 调整 groundtruth.txt 的 y 坐标 |

以上三个工具用 `--dataset-dir` 指定目录。标注更新工具会修改所选数据，应先核对路径与提示。

## 计数

```bash
python3 dataset_tools/count_saved_data.py --task normal --index 5
python3 dataset_tools/count_saved_data.py --task normal --index 5 --count 3
python3 dataset_tools/count_saved_data.py --task normal --index 5 --auto
python3 dataset_tools/count_saved_data.py --task normal --all
```

默认只统计第 0 组；`--record-dir` 可指定目录。`--all/--auto/--count` 互斥，`--all` 不能同时指定 index。
`--auto` 遇首个缺失配对停止；count/all 报告缺失后继续。RGB 统计实际帧数，不乘 ratio；事件包数依赖 delta-t（默认 2000 μs）。
