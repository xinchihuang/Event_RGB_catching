# Event RGB Catching 修改台账

记录采集、在线处理、离线预处理、文件回放和评估工具的审查问题及修复状态。每次完成修改后，必须同步更新本表和下方修改记录。

建立日期：2026-10-11。最近更新：2026-10-11（America/New_York）。

本表以最后一次区分采集与回放的整体审查为基线，沿用其中的 P1-01 至 P1-08、P2-01 至 P2-08。更早聊天中的六项 P1 编号不作为本表编号。首次登记不代表本次完成了代码修复。

2026-10-11 用户将 P1-03 本次处理范围明确为“每次批量测试创建新文件夹，保存本批测试数据”。该项按批量范围验收；原审查提到的直接单次运行及 manual 同编号覆盖行为未修改，保留为后续范围。

## 状态和更新规则

- **待修改**：问题仍存在，尚未完成对应修复。
- **部分完成**：部分入口或子项已处理，剩余范围必须写清楚。
- **已修改待验证**：代码已修改，但必要验证尚未完成。
- **已完成**：该项范围内的修改和必要验证均已完成；标明验证边界。
- **待验证**：属于兼容性或时序风险，需要进一步验证，不能当成已复现故障。
- **无需修改**：现有行为符合所属阶段的输入约定，保留原因。

P1 表示可能影响数据正确性、运行或配置生效；P2 主要涉及可读性、默认值、路径和文档一致性。

每次修改完成后，在同一轮工作中：

1. 更新对应编号的状态、已完成内容、剩余项、验证结果和最近更新日期。
2. 在修改记录中追加日期、编号、实际修改文件、修改内容及验证结论；不要覆盖历史记录。
3. 未覆盖的入口不能随其他入口一起标记完成；没有实际运行的测试不能写成通过。
4. 新发现的问题使用新编号；已有编号不重排、不复用。超出本表的独立修改也应登记。
5. 状态依据工作区实际代码，不以 Git 是否已提交、文件是否显示 M 判断。

## 数据阶段和旋转边界

| 阶段 | 入口 | 当前输入和输出约定 | 状态 |
| --- | --- | --- | --- |
| 采集并保存 | [collect_data.py](dataset_tools/collect_data.py) | RGB 旋转后保存；事件数组旋转用于预览，RAW 由设备日志接口录制，并非保存旋转后的数组。 | 旋转本身无需统一删除；保存问题见 P1-01、P1-02。 |
| 在线跟踪和接球 | [tracking.py](tracking.py)、[live_real.py](live_real.py) | 实时相机输入按照安装方向旋转并对齐，不读取 processed_data。 | 保留在线变换；固定安装变换与兼容键说明已修正，见 P2-01。 |
| 离线预处理 | [time_align_rewrite.py](dataset_tools/time_align_rewrite.py) | 输入原始事件，旋转、裁剪后保存；保存事件未做 RGB 空间对齐。 | 当前旋转符合回放输入约定。 |
| 文件回放 | [record.py](record.py)、[record_compare.py](record_compare.py)、[evaluation_batch.py](tools/evaluation_batch.py) | 输入已旋转的 processed_data，只做 RGB 对齐。 | 取消重复旋转已完成，见 DONE-01。 |
| 训练数据集生成 | [prepare_dataset.py](dataset_tools/prepare_dataset.py) | 当前按原始事件输入设计，执行旋转和对齐。 | 旋转不能直接认定为错误；文档与路径衔接见 P2-02、P2-03。 |

## 已完成事项

这里登记首次建表前已存在、并已核对的修复。核对日期不是代码最初修改日期。

| 编号 | 范围 | 状态 | 已完成内容 | 验证和边界 | 最近更新 |
| --- | --- | --- | --- | --- | --- |
| DONE-01 | 单次、批量及对比回放 | 已完成 | [公共读取器](data_input/read_from_aligned_buffers.py) 已删除旋转分支，只做 RGB 对齐；回放入口复用该读取器，独立于实时 apply_rotation。 | 调用链与 Git 差异已核对；旋转及入口测试 7 项通过。已补充 normal/0 的 Event 无界面真实回放；未核对预测精度或其他方法/场景。 | 2026-10-11 |
| DONE-02 | 旧事件文件读取器、DBSCAN 工具 | 已完成 | [read_from_raw.py](data_input/read_from_raw.py)、[interactive_dbscan_tuner.py](tools/interactive_dbscan_tuner.py) 已移除各自旋转逻辑，复用公共回放读取器。 | 调用链静态核对；公共读取器已有旋转测试。工具 GUI 未做实机验证。 | 2026-10-11 |

## P1 修改表

| 编号 | 范围 | 状态 | 问题与位置 | 已完成内容和剩余修改 | 验证记录 | 最近更新 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-01 | 采集保存 | 已修改待验证 | [collect_data.py](dataset_tools/collect_data.py)、[recording_session.py](dataset_tools/recording_session.py)：修复事件开始时、RGB 结束时分别读取可变编号的竞争。 | 已实现：共享会话固定本次编号，两侧成功关闭文件后确认保存；收到两侧确认才递增编号；录制及保存期间拒绝重复开始，保留迟到 worker 的录制请求。剩余：双相机实机确认连续录制文件配对。 | 新增自动化测试覆盖跨 spawn 进程延迟保存、重复/过期确认、快速结束后退出和编号 0；采集专项 10 项通过，全部 tools 测试 33 项通过。尚未接相机验证。 | 2026-10-11 |
| P1-02 | 采集保存 | 已修改待验证 | [collect_data.py](dataset_tools/collect_data.py)、[recording_session.py](dataset_tools/recording_session.py)：修复退出时直接终止写入进程的问题。 | 已实现：q、Ctrl+C、SIGTERM 请求保存后退出；RGB finally 刷新缓冲、RAW 停止日志后确认；共用 60 秒宽限期，超时再终止；空视频、writer 失败、worker 异常及未确认配对返回失败。剩余：实机确认 RAW 收尾、MP4 可读性和较长录制退出。 | 新增自动化测试覆盖退出中的 RGB 刷盘、相机异常后的保存、writer 打开失败、RAW stop 调用顺序、慢保存等待及超时强制退出；采集专项 10 项、全部 tools 测试 33 项通过。相机 SDK 使用替身，未做真实录制。 | 2026-10-11 |
| P1-03 | 批量回放输出 | 已完成 | [run_record.py](run_record.py)、[evaluation_batch.py](tools/evaluation_batch.py)：按用户本次要求，每次批量调用创建一个新目录，所有测试归入本批 results/。 | 新版批量原已实现；现已补齐旧 JSON 批量入口，增加配置快照、--output-root 和已有目录拒绝复用；子进程覆盖继承的 EVALUATION_RESULTS_DIR，父环境不变。验收限批次隔离；直接单次运行、manual 和同一批内重复方法/场景/编号的覆盖规则未改。 | 新增专项 3 项通过：多配置多样本共用目录、连续两批不覆盖、自定义目录保护和 CLI 传参；全部 tools 测试 36 项通过。测试模拟 record.py 输出，未运行真实数据推理。 | 2026-10-11 |
| P1-04 | 在线跟踪、对比回放 | 已修改待验证 | [tracking.py](tracking.py)、[live_real.py](live_real.py)、[record_compare.py](record_compare.py)：单个 worker 失败后的停止、清理和退出码。 | 三个入口接入公共 process_lifecycle：检测非零退出，广播 stop 和退出状态；统一等待 5 秒，再 terminate/kill 并 join；故障或强制停止返回 1。启动失败也清理；信号仅请求停止，由主流程统一收尾。对比回放 worker 失败不导出最终 CSV，保存报错仍释放共享内存；键盘和 starter 配合退出。record.py 保留已有监控实现。剩余：真实相机、电机和 GUI 环境验证。 | 新增 12 项自动化测试，含真实 spawn 故障注入、EOF 同时失败、启动失败、超时 kill、三个入口退出码与清理、保存异常、信号请求与 starter 停止；全部 tools 测试 48 项通过。硬件初始化和共享内存清理使用替身，未做实机故障测试。 | 2026-10-11 |
| P1-05 | 共享拟合逻辑 | 已完成 | [fitting_process.py](tracking/fitting/fitting_process.py)、[parabola_fitter.py](tracking/fitting/parabola_fitter.py)、[ransac_fit.py](tracking/fitting/ransac_fit.py)：接通 trajectory.gravity、inlier_threshold。 | worker 启动时从合并后的 YAML 传入 MonoRansacDrag；重力贯穿 RANSAC、带阻力优化、零阻力回退及落点预测；阈值控制三维内点筛选。保持默认 -9.81 m/s²、0.35 m，校验有限值和正阈值，补两套配置注释及说明。修改配置后重启生效；旧实验拟合器不在当前接入范围。 | 新增专项 6 项通过：两套 YAML 与 overlay 覆盖到实际 worker 初始化、自定义重力拟合和预测、阈值改变内点、零阻力回退、默认参数与实例隔离、非法参数；全部 tools 测试 54 项通过。使用合成三维轨迹，替换像素投影；未验证真实数据预测精度或电机。 | 2026-10-11 |
| P1-06 | 废弃参考帧工具 | 已完成 | 已删除的 dataset_tools/get_reference_rewrite.py 原导入不存在的 data_input.read_from_dat。 | 用户确认工具已废弃，按要求删除脚本并移除 dataset_tools/README.md 中的工具入口，不再修复该脚本。 | 仓库引用检查无运行代码调用；仅保留台账和 data_inventory/20261004_190202 历史盘点中的来源记录。文件删除与差异格式检查通过；未运行硬件或数据测试。 | 2026-10-11 |
| P1-07 | 历史评估工具 | 已完成 | [plot_old_model_new_gt.py](evaluation_tools/plot_old_model_new_gt.py) 的 dark、dark_clutter Excel 起始编号颠倒。 | 已改为 dark=300–399、dark_clutter=200–299，并同步导出的 method.txt 说明；保持 excel_id=场景偏移+local_id 单向相加，不取模、不循环映射。逐场景仍读取原标签文件及组内 0–99 号轨迹。 | 与标签说明、新批量及 Excel 对比工具映射一致；执行实际 excel_id 表达式验证四场景 400 个编号连续、唯一，验证不回绕；语法和 git diff --check 通过。未运行旧拟合器或重新生成历史输出。 | 2026-10-11 |
| P1-08 | 回放、计数工具 | 已完成 | [read_from_aligned_buffers.py](data_input/read_from_aligned_buffers.py) 覆盖 ratio；[count_saved_data.py](dataset_tools/count_saved_data.py) 强制 --all，其他分支 RGB 总帧数误乘 10。 | 回放 ratio 正整数校验、默认 10，移除强制覆盖；record、compare（含 All/Save）、shell、新旧批量入口支持 --ratio，旧 JSON 支持逐项 ratio 且 CLI 优先，批次保存生效值。计数默认 index=0/count=1，all/auto/count 互斥，all 拒绝 index；auto 缺配对即停，count 和 all 报告缺失后继续；修正 RGB 总数。仅文件回放与计数，不改实时采集。 | 新增专项 8 项及扩展批量测试通过；全部 tools 测试 62 项通过。模拟数据验证实际回放循环 ratio=2/3/10、裁剪、入口与子进程传参、JSON 覆盖、无效参数、缺失配对及实际帧总数；bash -n、git diff --check 通过。未使用真实录制或 GUI。 | 2026-10-11 |

## P2 修改表

| 编号 | 范围 | 状态 | 问题与位置 | 剩余修改和完成条件 | 验证记录 | 最近更新 |
| --- | --- | --- | --- | --- | --- | --- |
| P2-01 | 实时旋转说明 | 已完成 | 实时事件读取器固定旋转，旧文档却宣称 apply_rotation 控制实时旋转。 | 明确固定安装变换 (x,y)→(y,y_offset−x)，apply_rotation 为未使用的兼容键；更新两套配置说明、YAML 与读取器注释。保留在线算法和相机方向，回放仍只做 RGB 对齐。 | 调用链核对；78 项测试通过，包含回放不重复旋转；本项为语义与文档修正，未连接实时相机。 | 2026-10-11 |
| P2-02 | 数据流程文档 | 已完成 | [dataset_tools/README.md](dataset_tools/README.md) 原把预处理和训练数据集生成写成顺序步骤。 | 随 P2-03 路径接入明确两条流程：原始数据→预处理→回放，原始数据→训练集→合并/审核。注明事件旋转、RGB 对齐的执行阶段及预览/保存差别；明确 prepare_dataset 不接受已预处理回放输入。 | 与实际读取、保存和变换调用核对；路径专项验证各场景采集输出同时作为预处理与训练输入，预处理输出与回放输入一致。本次补充原始录制→新 DAT/MP4→回放单样本验证通过；训练生成流程未实跑。 | 2026-10-11 |
| P2-03 | 数据路径 | 已完成 | 采集、预处理、训练集生成、回放及评估工具原硬编码机器目录，历史绘图依赖固定旧源码路径。 | 增加 project_paths.py 与忽略 Git 的 config/paths.local.yaml，环境 EVENT_RGB_DATA_ROOT 覆盖本机根目录；共享 YAML 改用 ${data_root}，两套 Config 共用解析。各阶段 CLI 明确输入输出，采集显式传路径给 spawn worker；历史工具配置旧源码、轨迹和 Excel。相对路径按项目根目录；旧 DATA_ROOT 仍指 processed_data，精确路径与批次隔离保留，删除空格目录自动回退；缺失输入报错。配套合并、查看和标注工具也接入。 | 新增路径专项 11 项通过，全部 tools 测试 73 项通过；验证根目录切换、优先级、任意 cwd、两套配置、阶段衔接、重叠目录拒绝、历史目录校验、真实 spawn 输出路径、15 个工具 help；语法、bash -n、git diff --check 和本机配置忽略规则通过。未迁移数据；挂载后补充 normal/0 的 Event 无界面真实数据回放通过（退出码 0、生成 CSV、正常清理），未验证 Model/GPU、其他场景或采集。范围为本项目数据工具及运行入口，不包括 AMTTrack 独立训练工程的环境配置。 | 2026-10-11 |
| P2-04 | 预处理文件格式 | 已完成 | time_align_rewrite 原使用 DatWriter 却命名 .raw，并沿用旋转前尺寸。 | 新输出统一 .dat，旋转按原始宽度计算，DAT 头写旋转后的 H×W；写入函数拒绝 .raw 后缀。回放、shell、批量、计数和调参默认选 .dat，缺少时兼容历史 .raw；旧数据不迁移、不删除。 | 新增 5 项专项含真实 SDK DAT 写读、四角坐标/时间/极性保留、历史 .raw 兼容、文件选择及 Config 隔离；全部 78 项通过。真实 normal/0 原始录制→DAT/MP4→Event 无界面回放均退出 0，头部 1280×720，生成 34 条观测及 1 条落点。未验证其他场景、GUI 或预测精度。 | 2026-10-11 |
| P2-05 | 参数和数据语义 | 已完成 | 旧 impact_bounds_x/y、save_delay_sec、ball_pixel_3D 名称及部分参数注释容易误导。 | 保留兼容 YAML 键，明确 bounds_x 实际检查世界 Y、bounds_y 检查世界 Z（米、开区间）；局部变量改为 bounds_world_y/z。注明首次球观测后的墙钟延时、像素观测 u/v/size/event_time_s、世界 XYZ、重力 Z 方向及 distortion_coeffs→dist_coeffs 映射；标明未接入运行时的 timing/buffers 字段与历史拟合器范围。 | 核对实际写入、拟合、计时与边界检查；共享接口、原轨迹配置测试及全部 78 项通过；真实 Event 回放通过。保持配置值、保存格式和算法行为。 | 2026-10-11 |
| P2-06 | 默认值和提示 | 已完成 | 四个入口帮助中的 task/index 默认值错误；save_csv 提示未区分输出类型。 | 四个入口改用 argparse 自动展示声明默认值，保留既有默认行为；说明 save_csv 只控制依赖 GUI 的检测框 CSV，拟合轨迹/落点单独保存。同步 shell 提示、README 默认值表及各入口输出职责。 | 从 /tmp 执行 record、tracking、live_real、record_compare 的 --help 均退出 0；实际无 GUI 回放提示与输出一致；78 项测试通过。未启动 GUI。 | 2026-10-11 |
| P2-07 | 配置职责和优先级 | 已完成 | 两套 Config 重复且返回键有差异；实时配置选择和批量覆盖行为说明不完整。 | config_live.Config 继承共享访问实现，只绑定自己的进程内 loader；统一方法、返回键和缺省值。补明确 overlay 选择、CLI/环境/路径优先级、实时默认 config/ 与显式 config_live/；说明新批量快照强制 replay_apply_rotation=False、直接回放拒绝 True，旧 JSON 批量不改写。 | 共享接口与独立状态专项、严格配置与路径回归及全部 78 项通过；两套栈值不串用，真实 Event 回放通过。未改变批量现有覆盖规则。 | 2026-10-11 |
| P2-08 | 安装与工具文档 | 已完成 | 安装指南有旧解释器/路径，入口说明和 openpyxl 缺失，HSV 默认图片路径失效，存在废弃代码注释。 | 安装指南更新为解释器参数和本机路径配置，标明原版本表为历史部署记录；补回放/对比/新旧批量入口及输出说明，加入 openpyxl；HSV 默认 tracking/l.jpg 并按项目根目录解析；删除事件/FLIR/拟合流程中的废弃注释代码。run_live_real.sh 支持 VENV_DIR/PYTHON_BIN/SDK 目录环境覆盖并保留原默认值。 | HSV 从 /tmp 使用默认图片取样成功；四个主入口 help、Python 语法、shell 语法、差异格式检查通过；live shell 用假解释器验证参数与环境覆盖，未接相机/电机。全部 78 项通过；未在新机器重装整套依赖或重验历史版本组合。 | 2026-10-11 |

## 待验证事项

| 编号 | 范围 | 状态 | 风险和完成条件 | 最近更新 |
| --- | --- | --- | --- | --- |
| CHECK-01 | 在线及对比回放 GUI | 待验证 | tracking.py、live_real.py、record_compare.py 仍在子进程启动 GUI，未统一到 record.py 的主进程显示方式。需验证目标 OpenCV/Qt 环境中的启动、关闭和 worker 故障退出；不能仅凭静态代码认定一定卡住。 | 2026-10-11 |
| OPEN-01 | GitHub 公开准备 | 部分完成 | 已完成检查；当前源码/文档个人路径已清理，AMTTrack 训练路径可配置、旧视频读取器共用路径配置；.idea 的 8 个文件已从索引移除，本机文件不变；私人盘点、TensorBoard、评估导出加入忽略规则，原始台账备份保留本机。257 个可提交文本扫描无个人路径匹配，路径专项检查和 78 项回归通过。剩余：旧 Git 历史/提交身份、媒体/模板/标签/权重发布范围、许可及正式视频/Citation。本轮修改纳入本地提交，未推送或重写历史；见 [公开前检查](OPEN_SOURCE_REVIEW.md)。 | 2026-10-11 |

## 验证基线

- 建表前同一会话已运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p 'test_*.py'`：23 项通过。
- 最近一次旋转专项核对已运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.test_replay_rotation tools.test_processed_replay_entrypoints -v`：7 项通过。
- 这些测试不证明采集保存时序、真实相机、GUI 或完整批量评估通过。首次建表只保存已有核对结论，没有重新运行上述测试。
- 初次审查时 `${data_root}/processed_data` 不可访问；2026-10-11 用户挂载后已完成 normal/0 的 Event 无界面真实数据回放，详见后续验证记录。
- 2026-10-11 修改 P1-01、P1-02 后运行采集专项：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p 'test_capture_lifecycle.py' -v`，10 项通过；随后运行全部 tools 测试，33 项通过。采集测试使用模拟 SDK 和真实 spawn 子进程，不访问相机、不生成真实录制文件。
- 2026-10-11 修改 P1-03 后运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p 'test_batch_output_directories.py' -v`，3 项通过；随后运行全部 tools 测试，36 项通过。目录测试仅在临时目录模拟子进程 CSV 输出。

- 2026-10-11 修改 P1-04 后运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p 'test_*.py'`：48 项通过，其中停止流程新增 12 项。使用真实 spawn 子进程执行失败和强制停止；入口测试执行实际主流程收尾代码，替换硬件初始化和共享内存对象。未连接相机、电机或打开 GUI。

- 2026-10-11 修改 P1-05 后运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.test_trajectory_config -v`：6 项通过；随后运行全部 tools 测试：54 项通过。数值验证使用合成轨迹；配置接线测试执行实际 worker 构造代码，未启动硬件或验证真实场景精度。

- 2026-10-11 修改 P1-08 后运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.test_replay_options -v`：8 项通过；扩展批量传参测试后运行全部 tools 测试：62 项通过。模拟 SDK 输入和共享内存缓冲执行实际回放循环；未使用真实录制数据或 GUI。`bash -n run_record.sh`、`git diff --check` 通过。

- 2026-10-11 修改 P2-03 并同步 P2-02 后运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.test_project_paths -v`：11 项通过；全部 tools 测试 73 项通过。路径测试用临时目录与模拟写入器，spawn 为真实子进程；15 个工具从不同 cwd 执行 --help，通过且未打开硬件。Python 语法、`bash -n run_record.sh`、`git diff --check` 通过，`config/paths.local.yaml` 确认被 Git 忽略。

- 2026-10-11 挂载后运行 `bash ./run_record.sh --test_mode Event --task normal --record_index 0 --no-gui`：使用本机配置 `${data_root}`，读取 189 帧 RGB、1904 个事件包，对齐后为 1890 个事件包（ratio=10）。生成 `evaluation_tools/batch_runs/manual/Event_normal/ball_pixel_3D_records/0.csv`（1 行落点）与 `ball_pixel_3D_records_ball/0.csv`（34 行球观测）；退出码 0，5 个 worker 正常退出并清理共享内存。验证仅覆盖该样本文件回放的路径、处理、输出和正常退出，不代表预测精度、GUI、Model/GPU 或相机采集已验证。数据盘当前为只读挂载，结果写入项目目录。

- 2026-10-11 完成剩余 P2 修改：全部 tools 测试 **78 项通过**（新增 5 项包含真实 Metavision SDK 写读；本机未跳过）。四个入口从 /tmp 执行 --help、HSV 默认图片无 GUI 取样、live shell 假解释器环境覆盖、Python/shell 语法及 git diff --check 通过。
- 真实数据衔接验证输出：[p2_validation_9eh0w749](evaluation_tools/batch_runs/p2_validation_9eh0w749)。执行 `time_align_rewrite.py --task normal --index 0 --count 1 --no-view --save-dir <验证目录>/processed`，从原始录制生成 event/0.dat 与 rgb/0.mp4；再以 `run_record.sh --test_mode Event --task normal --record_index 0 --no-gui --data-dir <验证目录>/processed` 回放，输出重定向到验证目录 results。两步退出码均为 0，DAT 头部 H×W=1280×720，保存 34 条球观测与 1 条落点；5 个 worker 正常退出并清理共享内存。保留 preprocess.log、replay.log、tests.log。仅验证流程和格式，不声称预测精度、Model/GPU、实时相机、电机或 GUI 已通过。

- 2026-10-11 文档精简：集中公共配置说明，主 README 增加视频/Citation 占位；核对本地文档链接、锚点和命令参数，git diff --check 通过。本轮只修改文档，未重跑算法或硬件测试；78 项测试为前一轮结果。公开检查范围与剩余项见 OPEN-01。

- 2026-10-11 OPEN-01 路径清理：AMTTrack 训练配置默认目录、环境变量覆盖及任意 cwd 验证通过；旧视频读取器配置/显式目录在替身共享内存下检查通过。全部 tools 测试 78 项通过（9.362 秒）；257 个可提交工作区文本文件未发现个人绝对路径；.idea 的 8 个索引条目移除，本机文件哈希不变；忽略规则、Python 语法及 git diff --check 通过。未运行真实训练、相机或视频回放。

## 修改记录

按实际修改时间追加。已有修复首次登记时写“基线登记”，不要把登记日期当作修复日期。

| 日期 | 编号 | 修改文件 | 实际内容 | 验证结果与剩余项 |
| --- | --- | --- | --- | --- |
| 2026-10-11 | DONE-01、DONE-02 | 公共读取器、旧读取器、DBSCAN 工具及回放入口 | 基线登记：回放取消重复旋转及公共读取器复用已存在。 | 调用链已核对，7 项旋转及入口测试通过；硬件和真实数据未验证。 |
| 2026-10-11 | P1-03、P1-04 | run_record.sh、tools/evaluation_batch.py、record.py、live_real.py | 基线登记：新输出目录和部分入口 worker 故障检测已存在。 | 仅部分完成，剩余项见对应表格；未将整项标记为完成。 |
| 2026-10-11 | 台账建立 | REVIEW_TRACKER.md、AGENTS.md、README.md | 保存最新审查表、明确数据阶段和已完成项；增加每次修改后维护台账的仓库约定及 README 入口。 | 文档链接、编号、状态及差异格式检查通过；未修改运行代码。 |
| 2026-10-11 | P1-01、P1-02 | dataset_tools/collect_data.py、dataset_tools/recording_session.py、dataset_tools/README.md、tools/test_capture_lifecycle.py、REVIEW_TRACKER.md | 增加录制会话与双路保存确认；固定编号、禁止保存期间重启；退出时先保存再等待，超时才强制终止并报告失败；补使用说明和回归测试。目录创建移到 main，方便无硬件导入测试；未改采集旋转、回放逻辑和数据默认路径。 | 专项 10 项通过，完整 tools 测试 33 项通过，git diff --check 通过。两项状态改为已修改待验证；剩余为真实双相机录制、RAW/MP4 文件完整性验证。 |
| 2026-10-11 | P1-03 | run_record.py、tools/test_batch_output_directories.py、README.md、REVIEW_TRACKER.md | 按用户明确的批量范围补齐旧入口：每批创建时间戳目录和配置快照，统一本批所有子进程 results 路径；支持指定不存在的新目录并拒绝复用已有目录。新版批量已有同样隔离，无需修改其代码。 | 专项 3 项通过、全部 tools 测试 36 项通过、git diff --check 通过；状态按批量范围改为已完成。单次及同一批内重复编号行为保留，未进行真实数据推理。 |
| 2026-10-11 | P1-04 | process_lifecycle.py、tracking.py、live_real.py、record_compare.py、tools/test_process_lifecycle.py、README.md、REVIEW_TRACKER.md | 补齐 worker 失败检测和失败退出码；统一协作停止、超时 terminate/kill、join 后释放共享内存；覆盖启动失败清理，避免信号处理重复收尾；对比回放失败不导出残缺 CSV，保存异常仍清理；修复 tracking 键盘等待和 compare starter 在停止后的状态改写。 | 新增停止流程 12 项测试通过，全部 tools 测试 48 项通过，git diff --check 通过。状态改为已修改待验证；剩余真实相机、电机及 OpenCV/Qt 故障退出验证，CHECK-01 仍待验证。 |
| 2026-10-11 | P1-05 | tracking/fitting/fitting_process.py、parabola_fitter.py、ransac_fit.py；config/ 与 config_live/ 的 tracking_config.yaml、README.md；tools/test_trajectory_config.py、REVIEW_TRACKER.md | 将 YAML 重力和内点阈值传入当前拟合器，全链路显式传递重力，避免不同拟合实例共享可变重力；保持旧默认值，补单位、方向、生效时机及参数校验。 | 专项 6 项、全部 tools 测试 54 项通过，git diff --check 通过；按配置接入范围标记已完成。没有运行真实相机、电机或真实数据精度评估。 |
| 2026-10-11 | P1-06 | dataset_tools/get_reference_rewrite.py（删除）、dataset_tools/README.md、REVIEW_TRACKER.md | 用户确认参考帧工具已废弃；删除脚本及 README 入口，关闭缺失模块导入问题；历史数据盘点中的来源记录保留。 | 检查删除结果、剩余引用及 git diff --check；无运行代码调用该工具，状态改为已完成。本次仅删除废弃工具和更新文档，未重跑自动化或硬件测试。 |
| 2026-10-11 | P1-07 | evaluation_tools/plot_old_model_new_gt.py、REVIEW_TRACKER.md | 修正 dark/d=300、dark_clutter/dc=200，同步 method.txt 范围说明；按用户要求保留单向偏移加法，禁止取模回绕。已有按 Excel ID 生成随机种子的公式保留，重新运行时种子会随修正后的 ID 改变。 | 实际编号表达式的 400 个映射、唯一性、不回绕、语法及差异格式检查通过；状态改为已完成。未重跑全部测试，未重拟合或覆盖现有历史图表、CSV。 |
| 2026-10-11 | P1-08 | replay_options.py、data_input/read_from_aligned_buffers.py、record.py、record_compare.py、run_record.sh、run_record.py、tools/evaluation_batch.py、dataset_tools/count_saved_data.py、README.md、dataset_tools/README.md、tools/test_replay_options.py、tools/test_evaluation_batch.py、tools/test_batch_output_directories.py、REVIEW_TRACKER.md | 接通正整数 ratio 到单次和批量回放，保留默认 10；批量记录生效比例，旧 JSON 支持逐项设置和 CLI 覆盖。计数默认指定单个编号，明确互斥和缺失配对处理，移除 RGB 帧数倍乘；SDK 导入移到实际读取处，帮助和选项验证不依赖 SDK。 | 新增专项 8 项和扩展批量测试通过，全部 62 项测试通过，shell 语法及差异格式检查通过；两子项均完成。未做真实数据及 GUI 端到端验证，实时采集未改。 |
| 2026-10-11 | P2-03、P2-02 | project_paths.py、config/paths.local.example.yaml、本机 paths.local.yaml（忽略 Git）、.gitignore；两套配置 YAML/loader；collect_data、time_align_rewrite、prepare_dataset、count_saved_data、pick_positive、dataset_viewer、update_absent_by_flags、adjust_groundtruth_y；record、record_compare、tracking、live_real、run_record.py/.sh；evaluation_batch、interactive_dbscan_tuner、plot_batch_red_green；历史与标准评估绘图入口；README、两套配置 README、dataset_tools/README、路径测试及相关既有测试、REVIEW_TRACKER.md | 统一多机器数据根目录和阶段路径，保留 CLI/环境覆盖与批量输出规则；采集路径显式跨 spawn 传递；历史资源显式指定；入口检查并打印解析路径，工具帮助不提前导入旧工程或 SDK。同步修正训练生成与预处理的文档关系。当前机器根目录设置为 ${data_root}；没有复制、移动或删除已有数据。 | 新增专项 11 项、全套 73 项通过，语法和差异格式检查通过；P2-03、P2-02 按路径与文档范围标记已完成。真实相机、GPU、历史拟合、数据完整性未验证，P1-01/02/04 与 CHECK-01 的实机待验证状态保留。AMTTrack 独立训练配置不在本次范围。 |
| 2026-10-11 | P2-03 运行验证 | REVIEW_TRACKER.md | 用户挂载数据盘后，沿用现有路径配置执行 normal/0 的 Event 无界面回放；仅更新验证记录，未改运行代码。 | 实际读取数据、生成 1 行落点和 34 行球观测 CSV，退出码 0，全部 worker 正常退出并清理共享内存。其余场景、GUI、Model/GPU、采集及预测精度未验证；原有硬件待验证状态保留。 |
| 2026-10-11 | P2-01 | data_input/read_from_event.py；两套 camera_config.yaml、README.md；根 README.md、REVIEW_TRACKER.md | 明确固定安装旋转、apply_rotation 未使用及采集/回放边界，保持在线变换。 | 静态调用链及回归测试通过；不改变 P1 硬件待验证状态。 |
| 2026-10-11 | P2-02、P2-03 复核 | dataset_tools/README.md、README.md、REVIEW_TRACKER.md | 原已完成项继续保持已完成；同步 DAT 输出说明，并补充本机路径下的真实原始数据预处理→回放验证。 | normal/0 两阶段正常退出；未修改原数据，未实跑训练集生成。 |
| 2026-10-11 | P2-04 | dataset_tools/time_align_rewrite.py、data_input/read_from_aligned_buffers.py、run_record.sh、tools/evaluation_batch.py、tools/interactive_dbscan_tuner.py、tools/test_preprocessed_dat.py、相关 README、REVIEW_TRACKER.md | 新输出 DAT 后缀和旋转后尺寸，统一 DAT 优先及旧 RAW 兼容；不根据扩展名推断旋转。 | 新专项 5 项通过（含 SDK），全套 78 项通过；真实原始数据预处理、新格式 Event 回放均退出 0，原数据未改动。 |
| 2026-10-11 | P2-05 | 两套 tracking_config.yaml、配置文档；config_live/camera_config.yaml；tracking/fitting/fitting_process.py、parabola_fitter.py、shared_memory_manager.py、utils.py、REVIEW_TRACKER.md | 保留旧键，明确世界 Y/Z 边界、时间基准、观测格式和单位；更正默认球径说明、区分当前与历史拟合器，标明未消费配置字段。 | 配置、轨迹回归及真实 Event 回放通过；数值参数与文件格式保持兼容。 |
| 2026-10-11 | P2-06 | record.py、tracking.py、live_real.py、record_compare.py、run_record.sh、README.md、REVIEW_TRACKER.md | argparse 自动展示默认值；明确 GUI 检测框 CSV 与观测/落点 CSV 的独立控制。 | 四个实际入口从任意 cwd 执行 help 通过；无 GUI 回放正常保存轨迹与落点。 |
| 2026-10-11 | P2-07 | config/config_loader.py、config_live/config_loader.py、两套 README.md、tools/test_preprocessed_dat.py、REVIEW_TRACKER.md | 共用 Config 访问方法，保留独立 loader；记录 overlay、CLI、环境、路径及批量快照覆盖规则。 | 两栈接口/状态隔离、严格配置与路径测试通过，全套 78 项通过。 |
| 2026-10-11 | P2-08 | UBUNTU22_INSTALL.md、README.md、tools/README.md、requirement.txt、tools/hsv_selector.py、run_live_real.sh、data_input/read_from_event.py、read_from_flir.py、read_from_raw.py、tracking/fitting/fitting_process.py、REVIEW_TRACKER.md | 更新安装和入口示例、openpyxl、HSV 资源路径、live 解释器覆盖；清理废弃代码注释。 | HSV 默认图取样、四入口 help、live 假解释器覆盖、语法及差异检查通过；未安装新环境或启动硬件。 |
| 2026-10-11 | P2-02、P2-06、P2-07、P2-08 文档整理 | README.md、config/README.md、config_live/README.md、dataset_tools/README.md、tools/README.md、UBUNTU22_INSTALL.md、REVIEW_TRACKER.md | 基于当日修改重写精简中文说明；公共参数只保留一份、合并重复批量说明，保留数据阶段/格式/停止/输出边界和实机验证限制；安装详细部署记录保留，并给版本占位参数补引号以避免被 shell 识别为重定向。新增视频及 Citation 占位，不编造论文信息。 | 文档链接与锚点、命令参数和差异格式检查通过；未改运行代码，未重复执行算法测试。原 P1/P2 编号、状态和历史保留。 |
| 2026-10-11 | OPEN-01 | OPEN_SOURCE_REVIEW.md、README.md、REVIEW_TRACKER.md | 完成公开前检查，登记路径、媒体、IDE、忽略规则、研究数据和历史风险；常见密钥规则未命中，记录扫描限制。 | 审查含 230 个索引路径、99 个本地可达提交、774 个历史文本 blob、221 个索引文本 blob 及 ZIP 文本；公开清理待执行，本次不做推送、取消跟踪或历史重写。 |
| 2026-10-11 | OPEN-01 个人路径与 IDE 清理 | AMTTrack/lib/train/admin/local.py、lib/vis/visdom_cus.py、tracking/analysis_results.py；data_input/read_from_video.py、record.py、record_compare.py；.gitignore；config/README.md、evaluation_tools/labels_12162025_README.txt、OPEN_SOURCE_REVIEW.md、REVIEW_TRACKER.md；.idea 索引条目 | 训练路径改为项目/主目录派生及环境覆盖，视频默认路径接入统一配置，删除含个人路径的废弃注释。标签和台账脱敏；台账原文备份于被忽略的 data_inventory/private_review/，仅替换历史中的路径表示，保留原结论。私人来源产物保留本机并忽略；git rm --cached 移除 8 个 IDE 条目，文件未删除。 | 路径检查、78 项回归、257 个公开候选文本扫描、IDE 哈希及忽略规则、语法和差异检查通过；OPEN-01 标记部分完成。没有改写旧提交、提交或推送；媒体与许可等剩余项继续保留。 |
| 2026-10-11 | 本地提交整理 | OPEN_SOURCE_REVIEW.md、REVIEW_TRACKER.md 及本轮代码/文档/测试 | 按用户要求整理本地 Git 提交，包含个人路径清理与 .idea 取消跟踪；16 个未跟踪的实验 CSV 保留本机、不纳入提交。同步公开检查中的提交范围说明，不改变问题完成状态。 | 最近一轮 78 项测试通过；本次仅调整记录和提交范围，未重跑运行测试。提交前检查暂存差异和个人路径；不推送、不改写历史，硬件与公开准备剩余事项保留。 |

后续记录格式：日期／问题编号／修改文件／实际修改内容／执行的验证及结果／剩余事项。状态发生回退时也要登记原因。
