# Event RGB Catching

Ball tracking, trajectory fitting, and catching with event and RGB cameras. Supports raw data capture, offline replay, and live control.

## Demo video

<!-- Replace with a public video link or a GitHub-accessible asset; check visual and audio content for personal information before publication. -->
**Demo video: coming soon.**

## Quick start

Run from the project root. On a new machine, follow the [Ubuntu installation guide](UBUNTU22_INSTALL.md), then configure the data path:

```bash
cp -n config/paths.local.example.yaml config/paths.local.yaml
# Set data_root in paths.local.yaml to this machine's data directory.
./run_record.sh --task normal --test_mode Event --record_index 0 --no-gui
```

`paths.local.yaml` is excluded from Git. Use `EVENT_RGB_DATA_ROOT=/path/to/data` for a temporary override.
Replay requires preprocessed input in `processed_data/saved_data_l/{event,rgb}`; missing directories produce an error without searching for other data.

`run_record.sh` uses `python3` from PATH by default; select an environment with `--python /path/to/python` or `PYTHON_BIN`.
[requirement.txt](requirement.txt) lists dependencies but is not a lockfile. Install Metavision and PySpin separately; Model mode requires CUDA, AMTTrack, and a compatible spconv installation.

## Entrypoints

| Entrypoint | Purpose | Default mode / task / index |
| --- | --- | --- |
| `run_record.sh` | Replay one recording, then exit | Event / normal / 0 |
| `record.py` | Python replay entrypoint with explicit input/output paths | Model / dark / 0 |
| `tracking.py` | Camera tracking and impact prediction, without motors | Model / dark_clutter / 0 |
| `live_real.py` | Camera tracking, prediction, and motor control | Model / dark_clutter / 0 |
| `record_compare.py` | Export and overlay trajectories from three methods | Playback / clutter / 9 |
| `start_evaluation.sh` | Evaluate a fixed batch of combinations, without GUI by default | See below |
| `run_record.py` | Legacy JSON-driven batch replay | `batch_config.json` |

Scenes: `normal=l`, `clutter=c`, `dark_clutter=dc`, `dark=d`.
Methods: `Model` combines model ROI and event tracking, `Event` uses event tracking, and `RGB` uses RGB tracking. Single replay also supports `Show`.

```bash
# Replay one recording with Model; omit --no-gui to show windows.
./run_record.sh --task dark --test_mode Model --record_index 0 --no-gui
# Camera tracking only: t starts, w waits, e exits.
python3 tracking.py --task normal --test_mode Event
# Live catching: this starts the cameras and motors.
python3 live_real.py --task dark_clutter --test_mode Model
# Export Model/RGB/Event trajectories, then replay their overlays.
python3 record_compare.py --task normal --record_index 0 --test_mode Save --output-root evaluation_tools/comparison_run
python3 record_compare.py --task normal --record_index 0 --test_mode Playback --output-root evaluation_tools/comparison_run
```

Use `ENABLE_GUI=0` with Python entrypoints, or `--no-gui` with `run_record.sh`.
Comparison mode `All` runs Save followed by Playback; `--playback_speed` controls playback speed.
Replay `--ratio` is the number of event packets per RGB frame: a positive integer, default `10`. It does not control capture or playback speed.

## Batch evaluation

```bash
bash start_evaluation.sh --dry-run       # Check configuration and input coverage only.
bash start_evaluation.sh --limit 1       # Run index 0 for each combination; requires CUDA.
bash start_evaluation.sh                 # Full batch without GUI.
./run_record.sh --batch                  # Same batch with GUI.
python3 run_record.py --batch_config batch_config.json  # Custom JSON combinations.
```

The new batch runner uses Model in all four scenes, RGB in normal/clutter, and Event in normal/dark: indices 0–99 per combination, 800 runs total.
Each run has a default timeout of 180 seconds. Missing inputs are skipped; errors/timeouts are recorded before continuing. Ctrl+C stops the batch and its child processes.
`--limit N` selects the first N indices per combination; `--timeout`, `--ratio`, and `--data-root` override their respective settings.

Both batch runners **create a new directory for each invocation**, under `evaluation_tools/batch_runs/` by default, with all results in that batch's `results/`.
`--output-root` must name a directory that does not yet exist. The new runner also saves `configs/`, `logs/`, `summary.csv`, and `status.json`;
`latest.json` points to the latest batch. Exit codes: 0 completed, 2 process/output errors, 130 interrupted.
The legacy JSON runner saves `batch_config.json`; entries may set `ratio` and `record_dir`, with CLI `--ratio` taking precedence.
Repeated method/scene/index combinations within one batch still use the same result filenames.

See the [tool documentation](tools/README.md#evaluation) for plotting and historical evaluation.

## Data and results

The two data flows are independent; see [capture and data tools](dataset_tools/README.md):

```text
Raw capture --> rotation and time trimming --> processed_data --> file replay
            --> training dataset generation (rotation/alignment) --> merging and labeling
```

New preprocessing outputs `.dat`; replay also accepts legacy processed `.raw`. If both exist for an index, `.dat` takes precedence.
Replay only aligns to RGB. The extension does not establish whether events have been rotated; raw camera recordings cannot be replayed directly.

| Output | Contents and location |
| --- | --- |
| Single shell replay | `evaluation_tools/batch_runs/manual/<method>_<scene>/`; repeated indices may overwrite results. Use `EVALUATION_RESULTS_DIR` to separate runs. |
| Python replay/tracking without motors | `${data_root}/test_data/evaluation_results/<method>_<scene>/` by default |
| `ball_pixel_3D_records_ball/<id>.csv` | Pixel observations `(u, v, size, event_time_s)`, without a header |
| `ball_pixel_3D_records/<id>.csv` | Predicted impact `(X, Y, Z)` in meters, without a header; catching plane X=0 |
| `--save_csv` | Controls only GUI bounding-box CSV output in the working directory's `bbox_detections/`; unavailable without GUI and independent of trajectory/impact output |
| Comparison replay | `${data_root}/demo` by default, or the trajectory directory selected by `--output-root` |

`live_real.py` uses predictions to trigger motors and does not save the fitting worker's trajectory/impact CSVs described above.
An all-zero prediction alone does not prove that no ball was detected; fitting or bounds filtering may also produce it.

Live/comparison entrypoints coordinate shutdown on exit requests or worker failures: wait 5 seconds, terminate/kill remaining processes, then release shared memory.
Normal exit returns 0; failure or forced shutdown returns 1. Comparison worker failure prevents final CSV export. Capture uses a separate 60-second save timeout.

## Documentation and validation

| Document | Scope |
| --- | --- |
| [Configuration](config/README.md) | Paths, precedence, parameters, rotation, and coordinates |
| [Live configuration](config_live/README.md) | Selecting independent live tuning settings |
| [Data tools](dataset_tools/README.md) | Capture, preprocessing, training data, and counting |
| [Tuning and evaluation](tools/README.md) | HSV, DBSCAN, plotting, and label mapping |
| [Installation guide](UBUNTU22_INSTALL.md) | Ubuntu deployment record and SDK/dependency troubleshooting |
| [Review ledger](REVIEW_TRACKER.md) | P1/P2 status, validation scope, and full history |
| [Maintenance instructions](AGENTS.md) | Recording and verifying changes |
| [Public release review](OPEN_SOURCE_REVIEW.md) | Sensitive information, tracked files, and historical content |

2026-10-11: P2 is complete, with 78 tests passing. One real sample passed raw recording → DAT/MP4 → headless Event replay.
Capture saving, live fault shutdown, and GUI behavior still require hardware validation; these checks do not establish prediction accuracy or coverage of all scenes.

## Citation

<!-- Add the actual paper title, authors, venue, year, DOI/arXiv link, and BibTeX after publication. Do not add a fabricated citation. -->
**Paper link and BibTeX: coming soon.**
