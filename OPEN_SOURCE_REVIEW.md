# GitHub 公开前检查

检查及更新日期：2026-10-11。已清理当前源码/文档的个人绝对路径，并将 `.idea/` 从索引移除；本机 IDE 文件保留。本轮修改以本地 Git 提交保存，不推送或重写 Git 历史。
README 已预留 Demo video、Citation；正式视频、论文链接和 BibTeX 尚待提供。

## 检查结论

常见私钥、GitHub/AWS/API 令牌、凭证赋值及含密码 URL 的规则扫描未命中。
随后已清理当前工作区中的个人路径，含路径的私人盘点/评估产物保留本机并加入忽略规则。当前可提交文本扫描无个人路径匹配项；历史、媒体和许可仍需单独处理。本地提交包含清理后的源码、文档和测试；16 个未跟踪的实验标签/失败列表 CSV 保留本机，不纳入本次提交。

| 内容 | 发现位置 | Git 状态 / 建议 |
| --- | --- | --- |
| 本机数据配置 | `config/paths.local.yaml` | 已忽略；只提交 paths.local.example.yaml |
| 本机训练路径 | `AMTTrack/lib/train/admin/local.py` | 已改为 AMTTrack 根目录、用户主目录与 AMTTRACK_* 环境变量解析；默认保持本机原目录含义 |
| 残留个人路径 | `AMTTrack/lib/vis/visdom_cus.py`、`AMTTrack/tracking/analysis_results.py`、`data_input/read_from_video.py`、`record.py`、`record_compare.py` | 已清理；视频读取器使用统一路径配置，其余删除废弃注释中的路径 |
| IDE 配置 | `.idea/` 下原有 8 个跟踪文件 | 已执行 git rm --cached，并加入忽略规则；本机文件内容未变 |
| 本地盘点与日志 | `data_inventory/`、`AMTTrack/tensorboard/`、`bbox_detections/` | 已加入忽略规则并保留本机；盘点含来源路径、时间等，不对原始实验记录做批量改写 |
| 评估导出及打包文件 | `evaluation_tools/landing_error_boxplot/provenance.json`、`sample_errors.csv`、`landing_error_boxplot_bundle.zip` | 已忽略目录和 ZIP，原始导出保留本机；如果后续要发布这些产物，需要单独脱敏 ZIP 内外的来源路径 |
| 标签来源及维护记录 | `evaluation_tools/labels_12162025_README.txt`、`REVIEW_TRACKER.md` | 已脱敏：标签仅保留工作簿名，台账使用 ${data_root}；原始台账备份保留在已忽略的 data_inventory/private_review/ 中 |
| 视频和场景模板 | 已跟踪 `AMTTrack/tracking/output_video.mp4`、`tracking/{l,c,d,dc}.{jpg,npy}` | 逐个确认画面、音轨和数据是否可公开；模板是 Model 运行资源，不能不提供替代方案就删除 |
| 实验标签与结果 | 已跟踪 `evaluation_tools/label_*.csv`、`indices.csv`，以及未跟踪的新标签/失败列表 | 属于研究数据，先决定发布范围；无需一概忽略全部 CSV |
| Git 提交身份与旧内容 | 本地可达历史含 3 个不同提交邮箱；历史 `doc/*.mov`、旧配置/脚本中有媒体和个人路径 | 确认身份公开范围；删除当前文件不删除历史，需要单独选择历史清理或干净发布目录方案 |

数据盘上的原始录制并不因配置引用就被 Git 上传；真正需要检查的是被添加到仓库的内容及 Git 历史。
拍摄视频应检查人物、屏幕、白板、设备标识、声音和位置线索。本次没有逐帧/逐音轨审核，也没有解读所有二进制数组。

## 已落实的忽略规则

本机路径配置、batch_runs、AMTTrack/output 和部分图片/权重后缀原已忽略。
本轮新增 `.idea/`、`data_inventory/`、`AMTTrack/tensorboard/`、`bbox_detections/`、
`evaluation_tools/landing_error_boxplot/` 及 `landing_error_boxplot_bundle.zip`。
这些目录和文件保留本机；已跟踪的 `.idea` 已另外从索引移除。

`.gitignore` 不会自动移除已有跟踪文件或 Git 历史；例如场景模板图片仍然已跟踪。
其余预防性规则（`.env*`、其他虚拟环境、私钥文件等）可在确定发布范围后补充。
不一概忽略 `*.csv`、`*.npy`，以免遗漏需要提供的运行资源或允许公开的数据。

## 发布材料

- 未找到仓库级或 AMTTrack 子目录中的 LICENSE 文件。公开前确定项目许可，并核对第三方源码、权重和 SDK 的分发范围；不要自动给第三方代码重新指定许可。
- Model 需要模板和 checkpoint。决定哪些资源随仓库提供、哪些另行下载，并补下载位置与使用条件。
- README 的视频与 Citation 保持明确占位，待实际公开视频/论文信息确定后再填写。
- 若发现真实凭证，先撤销或轮换，再清理当前文件与历史；仅删除文件不能使已泄露凭证失效。

## 检查范围

首次检查核对 230 个索引路径、当时 122 个未忽略未跟踪文件、99 个本地可达提交；
扫描 221 个索引文本 blob、774 个历史文本 blob，以及评估 ZIP 中 8 个可读文本条目。
内容规则扫描限单文件/条目 2 MiB 内且不含 NUL 的文本；较大对象仅检查名称和大小。
本机路径文件与忽略规则单独核对；未完整扫描所有忽略目录、远端独有分支、GitHub 附件、LFS、二进制媒体或压缩包之外的附件。
因此“未命中密钥规则”仅表示该范围内未发现，不是全面无敏感信息保证。

本轮清理后复查 257 个可提交工作区文本文件，未发现个人绝对路径匹配项。
AMTTrack 默认路径/环境覆盖/任意 cwd、旧视频读取器配置与显式路径均验证通过；
78 项回归测试通过。`.idea` 索引为空，本机文件哈希不变，私人产物忽略规则检查通过。
以上不代表旧提交或媒体已脱敏；删除历史内容需要另行处理。
