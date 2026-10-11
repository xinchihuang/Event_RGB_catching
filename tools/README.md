# Tuning and evaluation tools

Run commands from the project root. Tools do not automatically edit YAML; copy tuning results into the selected [configuration](../config/README.md).

## HSV sampling

```bash
python3 tools/hsv_selector.py                       # Defaults to tracking/l.jpg.
python3 tools/hsv_selector.py --source /path/to/frame.jpg
python3 tools/hsv_selector.py --source 0             # Camera input.
python3 tools/hsv_selector.py --x 320 --y 180         # Sample the default image without GUI.
```

Click a pixel to print HSV values and a suggested range; ESC exits. `--dh/--ds/--dv` set range widths.
Image paths are relative to the project root. Store results in the overlay's `rgb_param.hsv_lower/hsv_upper`.

## DBSCAN tuning

```bash
python3 tools/interactive_dbscan_tuner.py --task clutter
python3 tools/interactive_dbscan_tuner.py /path/to/processed/event/0.dat --task clutter --scale 0.8
```

Input must contain already-rotated processed events; the tool only aligns to RGB. It defaults to clutter index 0, preferring `.dat` with legacy `.raw` fallback.
`eps` is the neighborhood radius in pixels; `min_samples` is the minimum point count. Store them in `tracking.dbscan_eps/dbscan_min_samples`.
`--scale` affects display only; `delta_t` controls event packets, while n_events does not limit packet size in the current mode.

| Key | Action |
| --- | --- |
| SPACE | Play/pause |
| j / l, u / o | Back/forward 1 frame, back/forward 10 frames |
| h / E | First/last frame |
| e / q | Increase/decrease eps by 1 |
| d / a | Increase/decrease min_samples by 5 |
| r, p | Restore configuration values, print parameters |
| + / −, ESC | Adjust playback frame rate, exit |

<a id="evaluation"></a>
## Evaluation and plotting

See the [project README](../README.md#batch-evaluation) for single, batch, legacy batch, and comparison replay commands.
The new batch runner accepts `--plot-xlsx /path/to/ground_truth.xlsx` and generates red/green scatter plots plus `original_red_green/success_rates.csv` after completion.

| Tool | Input and purpose |
| --- | --- |
| `tools/plot_batch_red_green.py` | Plot batch results from a batch directory and Excel workbook |
| `evaluation_tools/plot_landing_prediction_simple.py`, `plot_landing_prediction_multi.py`, `plot_landing_error_boxplot.py` | Read saved impacts and coordinate labels for four scenes without rerunning detection/fitting; accept `--root-dir` |
| `evaluation_tools/plot_replay_vs_excel.py` | Compare replay results with Excel; select the workbook with `--xlsx`, `GROUND_TRUTH_XLSX`, or local `ground_truth_xlsx` |
| `evaluation_tools/analyze_ball_csvs.py` | Inspect saved ball observations |
| `evaluation_tools/plot_old_model_new_gt.py`, `run_legacy_plots.py` | Reproduce historical fitting and plotting; require historical source and trajectories |

Historical tools accept `--legacy-source-root`, `--tracks-dir`, and `--output-dir`.
The first two also support `LEGACY_SOURCE_ROOT/LEGACY_TRACKS_DIR` or local `legacy_source_root/legacy_tracks`.
Missing historical source raises an error rather than substituting the current fitter. Use each script's `--help` for full options.

Excel IDs use a one-way offset, without modulo: normal=0–99, clutter=100–199, dark_clutter=200–299, dark=300–399.
See [label notes](../evaluation_tools/labels_12162025_README.txt) for provenance and coordinates.
The new batch red/green plots retain the 0.25 m threshold and GT offset (−0.05,+0.95 m), reporting both success among valid predictions and success including failed samples.
`no_detection` denotes an all-zero result, which may also arise from fitting/bounds filtering; it does not establish that detection failed.
