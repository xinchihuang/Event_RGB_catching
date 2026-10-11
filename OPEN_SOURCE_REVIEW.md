# Public release review

Reviewed and updated: 2026-10-11. Personal absolute paths have been removed from current source/documentation, and `.idea/` has been removed from the index while preserving local IDE files. The cleanup is saved in a local Git commit; Git history has not been rewritten. The subsequent requested push failed because github.com could not be resolved.
The README includes Demo video and Citation placeholders; final video/paper links and BibTeX remain pending.

## Findings

Pattern scans found no common private keys, GitHub/AWS/API tokens, credential assignments, or password-bearing URLs.
Personal paths were then removed from current working files. Private inventory/evaluation artifacts containing paths remain local and are ignored. The candidate text scan found no remaining personal path matches; history, media, and licensing still require separate work. The local commit includes cleaned source, documentation, and tests; 16 untracked experiment label/failure-list CSVs remain local and were excluded.

| Content | Location | Git status / action |
| --- | --- | --- |
| Local data configuration | `config/paths.local.yaml` | Ignored; commit only paths.local.example.yaml |
| Local training paths | `AMTTrack/lib/train/admin/local.py` | Resolved from the AMTTrack root, user home, and AMTTRACK_* overrides; defaults preserve the original directory meanings |
| Personal path remnants | `AMTTrack/lib/vis/visdom_cus.py`, `AMTTrack/tracking/analysis_results.py`, `data_input/read_from_video.py`, `record.py`, `record_compare.py` | Cleaned; the video reader uses shared path configuration and obsolete path comments were removed |
| IDE configuration | Eight previously tracked `.idea/` files | Removed with git rm --cached and ignored; local contents unchanged |
| Local inventory/logs | `data_inventory/`, `AMTTrack/tensorboard/`, `bbox_detections/` | Ignored and retained locally; inventories include source paths and timestamps, so original experimental records were not bulk-rewritten |
| Evaluation exports/archive | `evaluation_tools/landing_error_boxplot/provenance.json`, `sample_errors.csv`, `landing_error_boxplot_bundle.zip` | Directory and ZIP ignored; original exports remain local. Publishing them requires sanitizing provenance both inside and outside the archive |
| Label provenance and maintenance records | `evaluation_tools/labels_12162025_README.txt`, `REVIEW_TRACKER.md` | Sanitized: label notes identify the source workbook without a local path, and the ledger uses ${data_root}. The original ledger backup remains in ignored data_inventory/private_review/ |
| Video and scene templates | Tracked `AMTTrack/tracking/output_video.mp4`, `tracking/{l,c,d,dc}.{jpg,npy}` | Review visual/audio/data content individually. Templates are Model runtime resources and need replacements if removed |
| Experiment labels/results | Tracked `evaluation_tools/label_*.csv`, `indices.csv`, and untracked new labels/failure lists | Research data: decide the release scope; do not indiscriminately ignore every CSV |
| Git identities and historical content | Local reachable history contains three distinct commit email addresses, historical `doc/*.mov`, and personal paths in old configurations/scripts | Confirm identity disclosure scope; deleting current files does not remove history. Choose historical cleanup or a clean release directory separately |

Referencing recordings on a data disk does not upload them through Git. Review actual tracked content and Git history.
Check videos for people, screens, whiteboards, device identifiers, voices, and location clues. This review did not inspect every frame/audio track or decode all binary arrays.

## Applied ignore rules

Local path configuration, batch_runs, AMTTrack/output, and some image/checkpoint extensions were already ignored.
The cleanup added `.idea/`, `data_inventory/`, `AMTTrack/tensorboard/`, `bbox_detections/`,
`evaluation_tools/landing_error_boxplot/`, and `landing_error_boxplot_bundle.zip`.
These remain local; previously tracked `.idea` files were separately removed from the index.

`.gitignore` does not remove tracked files or history; scene template images, for example, remain tracked.
Additional preventive patterns (`.env*`, other virtual environments, private-key files) can be added once release scope is decided.
Do not ignore all `*.csv` or `*.npy`, which could omit required resources or approved data.

## Release materials

- No repository-level or AMTTrack LICENSE file was found. Choose the project license and check distribution rights for third-party source, weights, and SDKs; do not automatically relicense third-party code.
- Model mode requires templates and a checkpoint. Decide which resources ship with the repository and which require separate downloads, then document locations and terms.
- Keep the README video and Citation placeholders until actual public video/paper information is available.
- If real credentials are discovered, revoke or rotate them before cleaning current files and history; deleting a file does not invalidate exposed credentials.

## Review scope

The initial review inspected 230 indexed paths, 122 then-unignored untracked files, and 99 locally reachable commits;
scans covered 221 indexed text blobs, 774 historical text blobs, and eight readable entries in the evaluation ZIP.
Content scanning was limited to entries/files up to 2 MiB without NUL bytes; larger objects were checked only by name and size.
Local configuration and ignore rules were checked separately. Ignored directories, remote-only branches, GitHub attachments, LFS, binary media, and other archives were not exhaustively scanned.
No credential-pattern matches within this scope is not a guarantee that every artifact is free of sensitive content.

After cleanup, 257 candidate working-tree text files were checked with no personal absolute-path matches.
AMTTrack defaults/environment overrides/arbitrary working directories and the legacy video reader's configured/explicit paths were verified.
All 78 regression tests passed. The `.idea` index was empty, local IDE hashes were unchanged, and private-artifact ignore rules passed checks.
These results do not establish that historical commits or media are sanitized; historical content requires separate work.
