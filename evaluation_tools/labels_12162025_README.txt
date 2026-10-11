Source: the 2025-12-16 workbook containing 400 ball throws, Sheet1. Select its local path with ground_truth_xlsx or --xlsx; no workbook is bundled here.
Original columns: image_id,x_m,y_m; coordinates are in meters.
Output CSVs have no header and contain original x_m,y_m, without offsets, scaling, or coordinate transforms.
normal.csv: l, original IDs 0–99
clutter.csv: c, original IDs 100–199
dark.csv: d, original IDs 300–399
dark_clutter.csv: dc, original IDs 200–299
Each file has 100 rows: row 1 maps to local ID 0, row 100 to local ID 99.
Excel has 399 records and is missing image_id=12; row 13 of normal.csv is nan,nan and is skipped by plotting scripts.
Existing label_l/d/c/dc.csv files contain legacy grid-index labels and are retained.
Current plot_landing_prediction_multi/simple.py interprets the columns as impact-plane y,z and adds (-0.05,+0.95) meters. Confirm that the workbook axes match this convention before use.
These CSVs are located in evaluation_tools when available locally; plotting accepts labels through options such as --label-file normal=/path/to/labels.csv.
