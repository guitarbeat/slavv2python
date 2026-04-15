# Changelog

This file summarizes notable repository changes for the SLAVV Python port.

The repository does not currently use git tags or published release entries, so
the notes below describe recent development work rather than formal release
cuts.

## Unreleased

Recent work landed between 2026-03-21 and 2026-04-14.

### Added

- **Parity Workflow Completion** (Planned):
  - CLI reuse eligibility summaries that display safe rerun commands, missing artifacts, and recommended next actions after each comparison run.
  - Stage-isolated network gate reliability enhancements with fast-fail validation, timing measurement (<30s target), and execution metadata persistence.
  - Shared-neighborhood diagnostic integration that generates actionable reports identifying claim ordering differences, branch invalidation differences, and partner choice divergences between MATLAB and Python.
  - Proof artifact promotion system that maintains evidence of exact network parity achievement, including full provenance tracking, input/output fingerprints, and a maintained proof artifact index.
  - New CLI command `slavv parity-proof --run-dir <path>` to display latest proof artifact summaries.
  - Enhanced staged layout with new artifacts under `03_Analysis/` (diagnostics, proof artifacts) and `99_Metadata/` (loop assessment, network gate validation/execution).
- Lightweight MATLAB health-check support for the parity comparison CLI,
  including persisted `matlab_health_check.json` metadata under staged run
  roots.
- Workflow-assessment reports that classify staged comparison roots as
  reusable, analysis-ready, blocked, or requiring a fresh MATLAB run.
- Cached output-root preflight and MATLAB-status inspection for repeated parity
  reuse loops.
- Canonical reference docs for energy-method selection and for adding new
  extraction algorithms to the validated CLI/pipeline surface.
- Release-verification notes for the canonical April 13 MATLAB/Python
  comparison run, including preserved timing and parity findings.
- Stricter energy rejection criteria for watershed-based edge candidates in `source/slavv/core/tracing.py`, aligning Python's candidate generation more closely with MATLAB's restrictive requirements.
- Regression coverage in `dev/tests/unit/core/test_watershed_supplement_regression.py` to verify watershed supplement rejection rules and prevent future regressions.
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
- Workspace-local maintenance scripts and tooling snapshots grouped under
  `dev/scripts/maintenance/` and `dev/reports/tooling/`.
- Targeted regression coverage for parity-mode edge cleanup tie-breaking and
  shared fresh/resumable MATLAB-shaped strand construction.

### Changed

- Root documentation now points to the active Shared Neighborhood chapter, the
  maintained docs index, and `BOTTLENECK_TODO.md` as the parity backlog instead
  of older historical chapter entry points or missing TODO links.
- The direct Hessian energy path now uses the same lower-memory helper as the
  resumable energy path, keeping those two execution modes aligned.
- The Python pipeline now writes structured run metadata and checkpoints by
  default when running through resumable entry points.
- The Streamlit app now keys structured run directories by both uploaded input
  content and validated parameters, preventing stale checkpoint reuse across
  parameter changes.
- `slavv analyze` now reconstructs the topology it needs from the standard
  exported `network.json` before computing summary statistics.
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
- `make`, `make.ps1`, and CI now treat repo-root `python -m mypy` as the
  supported typecheck gate.
- The repo-root `python -m mypy` gate now also covers the share-report, web
  app, and run-state entrypoints in addition to the existing CLI/core surface.
- The repo-root `python -m mypy` gate now also covers
  `source/slavv/analysis/geometry.py`.
- CI now includes a Windows CLI-security lane and an app-enabled Ubuntu UI
  lane without expanding the full matrix.

### Fixed

- Removed the stale duplicated Hessian-energy branch that remained after the
  lower-memory helper became the maintained implementation.
- Eliminated ~1,000 invalid watershed-based edge candidates that were previously crossing background areas by enforcing stricter energy-sign checks.
- Resume guards no longer overwrite stored fingerprints before comparison, so
  changed inputs and parameters correctly block stale resumable runs instead of
  silently reusing old checkpoints.
- The Streamlit ML curation flow now accepts uploaded `.joblib` and `.pkl`
  models without requiring the files to exist on disk ahead of time.
- `slavv analyze` no longer fails on standard `network.json` exports that only
  contain the normal serialized vertex/edge surface.
- Standalone comparison "0 vertices" issue caused by directory pathing artifacts in the comparison script.
- Empty-network shape handling in exporters and visualization outputs.
- Evaluation app import issues around share-report functionality.
- UTF-8 launcher environment handling and staged run-info normalization.
- Manifest timing fallback behavior for staged comparison runs.
- Linux launcher-path test expectations.
- CI lint failures while restoring MATLAB shell-launcher coverage.
- Validation now preserves parity-sensitive parameters such as
  `comparison_exact_network` and `space_strel_apothem_edges`.
- MATLAB-energy tracing no longer fails primarily as a dangling-path problem;
  frontier runs now produce terminal candidates consistently.
- The end-to-end integration test now resolves the committed fixture from the
  repo-root `data/` directory.
- Stale `slavv-streamlit` and old mapping-doc references were removed from
  tests and contributor templates.
- Hessian direction estimation now explicitly pins the current scikit-image
  derivative behavior to avoid future-warning drift.
- Parity-mode edge cleanup now prefers shorter equal-energy duplicates before
  downstream MATLAB-style pruning.
- Parity-mode network construction now shares deterministic MATLAB-shaped
  strand assembly between fresh and resumable runs, including additive
  `strands_to_vertices` output for exact-network comparisons.

### Notes

- Commit `11f8445` on 2026-03-24 mostly expands test coverage for the MATLAB
  restart flow; the primary implementation work for restartable MATLAB
  comparison runs landed in `afed6e1` on 2026-03-23.
- Exact vertex parity remains established on the imported-MATLAB saved-batch
  surface.
- The best retained saved-batch closeout result is `110/110` vertices,
  `94/93` edges, and `49/54` strands.
- The stage-isolated MATLAB-edges-to-Python-network gate remains exact on the
  imported-MATLAB parity surface.
- The active post-Chapter-2 parity work is now focused on neighborhood-level
  claim ordering and branch invalidation rather than generic downstream network
  assembly.
- Stricter Phase 2 watershed gates (frontier reachability and per-origin caps) implemented in `source/slavv/core/tracing.py`.
- MATLAB-style best-first frontier tracing tightened around ordering and pruning semantics.
- Staged comparison layouts and metadata persistence (preflight, MATLAB status) verified across canonical reruns.

