来源：12162025丢球400次.xlsx（本机路径通过 ground_truth_xlsx 或 --xlsx 指定），Sheet1
原始列：image_id,x_m,y_m；坐标单位：米。
输出CSV无表头，两列依次为原始x_m,y_m；不做偏移、缩放或坐标变换。
normal.csv：l，原始编号0–99
clutter.csv：c，原始编号100–199
dark.csv：d，原始编号300–399
dark_clutter.csv：dc，原始编号200–299
每个文件固定100行，第1行对应组内编号0，第100行对应组内编号99。
Excel共399条记录，缺少image_id=12；normal.csv第13行为nan,nan，绘图脚本会跳过。
原有label_l/d/c/dc.csv为旧网格编号标签，本次保留。
当前plot_landing_prediction_multi/simple.py将两列解释为落点平面的y,z，并额外加偏移(-0.05,+0.95)米。使用前需确认Excel坐标轴与该约定一致。
这些CSV位于evaluation_tools内；绘图时可通过--label-file normal=文件路径等参数指定标签。
