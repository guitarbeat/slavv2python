# Changelog

This file summarizes notable repository changes for the SLAVV Python port.

The repository does not currently use git tags or published release entries, so
the notes below describe recent development work rather than formal release
cuts.

## Unreleased

Recent work landed between 2026-03-21 and 2026-03-24.

### Added

- File-backed run state for SLAVV processing, including stage snapshots,
  structured artifacts, progress events, ETA tracking, and fingerprint-based
  resume guards.
- Resume-aware pipeline execution for the energy, vertex, edge, and network
  stages, including persisted intermediate artifacts that allow interrupted
  runs to continue without restarting from scratch.
- CLI resume controls and inspection surfaces:
  - `slavv run --stop-after ...`
  - `slavv run --force-rerun-from ...`
  - `slavv status`
  - `slavv import-matlab`
- Streamlit run-status dashboard and UI controls for stopping early or forcing
  recalculation from a selected stage.
- Restartable MATLAB comparison workflow that resumes from the newest matching
  `batch_*` output for the same input.
- Shared run metadata for MATLAB/Python comparison tasks, including manifest and
  status output under staged run layouts.
- Repository-local agent workflow guidance in `AGENTS.md`.
- `slavv-app` launcher support and Python 3.12 CI updates.
- Share-report export support in the evaluation app.

### Changed

- The Python pipeline now writes structured run metadata and checkpoints by
  default when running through resumable entry points.
- MATLAB wrapper execution now runs one workflow stage at a time and records
  resume state in the output directory for safer reruns.
- Windows MATLAB launcher behavior now waits on the batch process and resolves
  the repo-root `external/Vectorization-Public` checkout.
- Comparison outputs are increasingly normalized around staged run folders such
  as `01_Input`, `02_Output`, `03_Analysis`, and `99_Metadata`.

### Fixed

- Empty-network shape handling in exporters and visualization outputs.
- Evaluation app import issues around share-report functionality.
- UTF-8 launcher environment handling and staged run-info normalization.
- Manifest timing fallback behavior for staged comparison runs.
- Linux launcher-path test expectations.
- CI lint failures while restoring MATLAB shell-launcher coverage.

### Notes

- Commit `11f8445` on 2026-03-24 mostly expands test coverage for the MATLAB
  restart flow; the primary implementation work for restartable MATLAB
  comparison runs landed in `afed6e1` on 2026-03-23.
