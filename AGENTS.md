# Repository working instructions

## Review ledger

- Before modifying this repository, read [REVIEW_TRACKER.md](REVIEW_TRACKER.md) and identify the issue ID, current status, and data stage.
- The user requires a record after every change. In the same turn, update the issue status, completed work, remaining work, verification results, and last-updated date, and append a change-history entry.
- Record independent changes outside the existing table as well. Preserve issue IDs and history; do not renumber, reuse IDs, or overwrite historical records.
- Mark fixes covering only some entrypoints as partial. Use "Changed; verification pending" when required validation remains. Never report unrun tests as passing or treat a Git commit as proof of completion.
- Label the first registration of an existing fix as a baseline entry; do not claim it was implemented in the current turn.
- Use English for repository documentation, comments, and docstrings. Keep existing identifiers, data formats, and external resource names compatible.

## Capture and replay boundaries

- Distinguish capture storage, live camera processing, offline preprocessing, file replay, and training dataset generation.
- Duplicate rotation has already been removed from processed_data replay: the shared reader only aligns events to RGB. Do not treat rotation in live or raw-data processing as an unfinished replay fix.
- The event preview array in collect_data and the device RAW logger are separate paths; RGB is rotated before saving.
- time_align_rewrite rotates raw events; prepare_dataset also expects raw events. Do not chain these stages or remove all rotation without checking their input contracts.
- Review live apply_rotation documentation separately from its implementation. Do not apply a False default in a way that changes the physical camera orientation.
