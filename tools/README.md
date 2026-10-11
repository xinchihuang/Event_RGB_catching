# 调参与评估工具

命令从项目根目录执行。工具不会自动改 YAML，调参结果需写入选中的 [配置文件](../config/README.md)。

## HSV 取样

```bash
python3 tools/hsv_selector.py                       # 默认 tracking/l.jpg
python3 tools/hsv_selector.py --source /path/to/frame.jpg
python3 tools/hsv_selector.py --source 0             # 相机
python3 tools/hsv_selector.py --x 320 --y 180         # 默认图片取样，不打开 GUI
```

点击像素输出 HSV 与建议区间，ESC 退出；`--dh/--ds/--dv` 控制区间范围。
图片路径相对项目根目录。结果填入 overlay 的 `rgb_param.hsv_lower/hsv_upper`。

## DBSCAN 调参

```bash
python3 tools/interactive_dbscan_tuner.py --task clutter
python3 tools/interactive_dbscan_tuner.py /path/to/processed/event/0.dat --task clutter --scale 0.8
```

输入必须是已旋转 processed 事件，工具只做 RGB 对齐。默认 clutter 场景第 0 组，优先 `.dat`，兼容旧 `.raw`。
`eps` 是邻域半径（像素），`min_samples` 是最小点数；写回 `tracking.dbscan_eps/dbscan_min_samples`。
`--scale` 只影响显示；`delta_t` 影响事件分包，当前模式下 n_events 不限制每包大小。

| 按键 | 操作 |
| --- | --- |
| SPACE | 播放/暂停 |
| j / l、u / o | 前后 1 帧、前后 10 帧 |
| h / E | 首帧/末帧 |
| e / q | eps 加/减 1 |
| d / a | min_samples 加/减 5 |
| r、p | 恢复配置值、打印参数 |
| + / −、ESC | 调整播放帧率、退出 |

<a id="evaluation"></a>
## 评估与绘图

单次、新旧批量及对比回放命令统一见 [项目 README](../README.md#批量评估)。
新批量支持 `--plot-xlsx /path/to/ground_truth.xlsx`，结束后输出红绿散点图与 `original_red_green/success_rates.csv`。

| 工具 | 输入与用途 |
| --- | --- |
| `tools/plot_batch_red_green.py` | 批次目录与 Excel，绘制批量结果 |
| `evaluation_tools/plot_landing_prediction_simple.py`、`plot_landing_prediction_multi.py`、`plot_landing_error_boxplot.py` | 读取已保存落点和四场景坐标标签，不重新检测/拟合；支持 `--root-dir` |
| `evaluation_tools/plot_replay_vs_excel.py` | 比较回放结果与 Excel；`--xlsx`、`GROUND_TRUTH_XLSX` 或本机配置 `ground_truth_xlsx` 选择工作簿 |
| `evaluation_tools/analyze_ball_csvs.py` | 检查保存的球观测 |
| `evaluation_tools/plot_old_model_new_gt.py`、`run_legacy_plots.py` | 复现历史拟合与绘图；需要旧源码和历史轨迹 |

历史工具用 `--legacy-source-root`、`--tracks-dir`、`--output-dir` 指定路径；
前两项也支持 `LEGACY_SOURCE_ROOT/LEGACY_TRACKS_DIR` 或本机配置 `legacy_source_root/legacy_tracks`。
旧源码缺失会报错，不替换成当前拟合器。完整选项以各脚本 `--help` 为准。

Excel 编号单向相加、不取模：normal=0–99、clutter=100–199、dark_clutter=200–299、dark=300–399。
标签来源及坐标见 [标签说明](../evaluation_tools/labels_12162025_README.txt)。
新批量红绿图沿用 0.25 m 阈值和 GT 偏移 (−0.05,+0.95 m)，同时报告有效预测成功率与计入失败样本的成功率。
`no_detection` 表示全零结果，也可能来自拟合/边界筛选，不等于确定未检测到球。
