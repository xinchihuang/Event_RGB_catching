# dataset_tools

Short notes on the dataset scripts and their typical flow.

## Typical pipeline
1) Capture raw data: `python3 dataset_tools/collect_data.py`
2) Time align/trim: `python3 dataset_tools/time_align_rewrite.py --record-dir ... --index ... --save-dir ...`
3) Build dataset frames: `python3 dataset_tools/prepare_dataset.py`
4) Merge/sample: `python3 dataset_tools/pick_positive.py`
5) Review/flag: `python3 dataset_tools/dataset_viewer.py`
6) Apply flags: `python3 dataset_tools/update_absent_by_flags.py`

## Scripts
- `collect_data.py`: live capture of FLIR RGB and event data (r=start, f=finish/save, q=quit).
- `prepare_dataset.py`: generate per-frame RGB/event/combined/voxel data and labels.
- `time_align_rewrite.py`: trim and align event packets to RGB frames; optional preview.
- `get_reference_rewrite.py`: export aligned event packs, RGB frames, and overlays for a record.
- `count_saved_data.py`: count event packets and RGB frames per record.
- `pick_positive.py`: merge datasets and sample negatives with a fixed probability.
- `dataset_viewer.py`: interactive viewer with RGB/Event/Combined/Voxel modes and flags.
- `update_absent_by_flags.py`: update absent.txt using frame_flags.txt.
- `adjust_groundtruth_y.py`: shift y in groundtruth.txt across a dataset.
