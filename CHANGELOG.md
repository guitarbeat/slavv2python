# Changelog

This file summarizes notable repository changes for the SLAVV Python port.

The repository does not currently use git tags or published release entries, so
the notes below describe recent development work rather than formal release
cuts.

## Unreleased

Recent work landed between 2026-03-21 and 2026-03-26.

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
- Real MATLAB HDF5 energy import for `slavv import-matlab`, producing
  pipeline-compatible checkpoints instead of placeholder energy payloads.
- A parity-only MATLAB-style frontier tracer for comparison runs that use
  MATLAB-origin energy and `comparison_exact_network`.
- Repository reference docs under `docs/`, including the refreshed MATLAB
  mapping and comparison layout guides.

### Changed

- The Python pipeline now writes structured run metadata and checkpoints by
  default when running through resumable entry points.
- MATLAB wrapper execution now runs one workflow stage at a time and records
  resume state in the output directory for safer reruns.
- Windows MATLAB launcher behavior now waits on the batch process and resolves
  the repo-root `external/Vectorization-Public` checkout.
- Comparison outputs are increasingly normalized around staged run folders such
  as `01_Input`, `02_Output`, `03_Analysis`, and `99_Metadata`.
- `slavv import-matlab` now prefers curated MATLAB vertices and edges when both
  curated and raw artifacts are present in a batch.
- Comparison summaries and reports now surface the actual Python energy source
  and frontier-specific tracing diagnostics during parity runs.

### Fixed

- Empty-network shape handling in exporters and visualization outputs.
- Evaluation app import issues around share-report functionality.
- UTF-8 launcher environment handling and staged run-info normalization.
- Manifest timing fallback behavior for staged comparison runs.
- Linux launcher-path test expectations.
- CI lint failures while restoring MATLAB shell-launcher coverage.
- Validation now preserves parity-sensitive parameters such as
  `comparison_exact_network` and `space_strel_apothem_edges`.
- MATLAB-energy tracing no longer fails primarily as a dangling-path problem;
  frontier runs now produce terminal candidates consistently, shifting the
  remaining gap into edge cleanup and strand construction.

### Notes

- Commit `11f8445` on 2026-03-24 mostly expands test coverage for the MATLAB
  restart flow; the primary implementation work for restartable MATLAB
  comparison runs landed in `afed6e1` on 2026-03-23.
- As of 2026-03-26, the MATLAB-energy control reaches exact vertex parity and
  improved edge parity (`1349` Python vs `1379` MATLAB), but exact edge and
  strand parity still remain open.
