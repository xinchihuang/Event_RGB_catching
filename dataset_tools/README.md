# Data capture and processing

Set data_root using the [path configuration](../config/README.md#paths). Run commands from the project root; `--task` selects normal/clutter/dark_clutter/dark.

## Two independent flows

| Stage | Input → output | Event transformation |
| --- | --- | --- |
| `collect_data.py` | Cameras → `saved_data_raw/saved_data_<scene>` | Device RAW logs are not rotated; event preview is separate from storage; RGB is rotated before saving |
| `time_align_rewrite.py` | Raw directory → `processed_data/saved_data_<scene>` | Rotation and time trimming; no RGB spatial alignment in saved events |
| File replay | Processed directory → tracking/impact results | RGB alignment only, without additional rotation |
| `prepare_dataset.py` | Raw directory → `saved_data/dataset_<scene>` | Performs its own rotation and RGB alignment to generate training frames/voxels/labels |

**Do not feed time_align_rewrite output into prepare_dataset.** Both read raw capture data.

## Capture and replay preprocessing

```bash
python3 dataset_tools/collect_data.py --task normal
python3 dataset_tools/time_align_rewrite.py --task normal --index 0 --count 1 --no-view
./run_record.sh --task normal --test_mode Event --record_index 0 --no-gui
```

Capture keys: `r` starts, `f` finishes and saves, `q` exits. Event and RGB share a fixed recording index; wait for `Saved pair` before starting again.
Repeated start requests during recording/saving are ignored. q, Ctrl+C, and SIGTERM request saving before exit and wait up to 60 seconds. Incomplete pairs, empty videos, and timeouts return failure.

New preprocessing outputs `event/<id>.dat` and `rgb/<id>.mp4`. Event coordinates become `(raw_y, raw_width−1−raw_x)`;
header height×width changes from 720×1280 to 1280×720, preserving timestamps and polarity. Preview RGB alignment is not written to the event file.
Legacy processed `.raw` files (including old DAT contents/dimensions) remain replayable. `.dat` takes precedence for the same index; old files are not deleted.

| Path option | Purpose |
| --- | --- |
| Capture `--save-dir` (alias `--output-dir`) | Raw output directory |
| Preprocessing `--record-dir / --save-dir` | Raw input and processed output |
| Training generation `--record-dir / --output-dir` | Raw input and training dataset output |

Preprocessing defaults to `index=0,count=20,ratio=10`. `--auto` processes from the starting index until the first missing pair; `--no-view` disables preview.
`--no-spatial-align` disables preview alignment only. Preprocessing and training generation reject identical or nested input/output directories.

## Training datasets and labels

```bash
python3 dataset_tools/prepare_dataset.py --task normal --index 0 --count 1
python3 dataset_tools/pick_positive.py
python3 dataset_tools/dataset_viewer.py
```

Training generation retains its preview behavior and defaults to indices 13–99 (`index=13,count=87`).
`pick_positive.py` merges all four scenes into `saved_data/selected_dataset` by default. Repeat `--source-dir` to select inputs and use `--output-dir` for output.

| Tool | Purpose |
| --- | --- |
| `dataset_viewer.py` | View RGB/Event/Combined/Voxel data and record flags |
| `update_absent_by_flags.py` | Update absent.txt from frame_flags.txt |
| `adjust_groundtruth_y.py` | Adjust y coordinates in groundtruth.txt |

These three tools accept `--dataset-dir`. Label-update tools modify the selected dataset; check paths and prompts first.

## Counting

```bash
python3 dataset_tools/count_saved_data.py --task normal --index 5
python3 dataset_tools/count_saved_data.py --task normal --index 5 --count 3
python3 dataset_tools/count_saved_data.py --task normal --index 5 --auto
python3 dataset_tools/count_saved_data.py --task normal --all
```

Only index 0 is counted by default; `--record-dir` selects a directory. `--all/--auto/--count` are mutually exclusive, and `--all` cannot be combined with an explicit index.
`--auto` stops at the first missing pair; count/all report missing inputs and continue. RGB counts actual frames without multiplying by ratio; event packet counts depend on delta-t (default 2000 μs).
