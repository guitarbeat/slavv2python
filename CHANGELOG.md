# Changelog

This file summarizes notable repository changes for the maintained SLAVV Python
codebase.

The repository does not currently use tags or formal release notes, so this
file stays intentionally lightweight. It highlights the current product surface
and recent changes without trying to preserve superseded workflow plans as if
they were still active.

For current behavior and proof status, prefer:

- [README.md](README.md)
- [AGENTS.md](AGENTS.md)
- [docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [docs/reference/core/EXACT_PROOF_FINDINGS.md](docs/reference/core/EXACT_PROOF_FINDINGS.md)

## [Unreleased]

### Added

- **New Analysis Metrics**: Extended network graph computations to include
  volume, surface area, densities, and edge energy statistics.
- **Paper-First Workflow Surface**: Added first-class `paper` and
  `matlab_compat` pipeline profiles across the CLI and Streamlit app, with
  `paper` as the public default.
- **Authoritative Network JSON**: `network.json` is now a versioned export that
  carries metadata, validated parameters, vertices, edges, topology, and
  optional precomputed summary statistics for direct `analyze` and `plot`
  consumption.

### Changed

- **Public Claim Boundary**: The maintained docs now distinguish the
  paper-complete native Python workflow from the developer-only exact MATLAB
  parity track.
- **Removed Deprecated Files**: Deleted legacy non-parity helper modules to
  simplify the edge extraction package around the maintained routes.
- **Refactored Core Pipeline**: Major overhaul of `process_image` and
  `calculate_energy_field` into modular helpers to improve readability and
  chunked processing.
- **Improved MATLAB Bridge**: Decomposed `import_matlab_batch` into focused
  helpers for better error handling and clearer stage resolution.
- **Enhanced Curator Logic**: Refactored automatic, Drew's, and ML curators to
  use reusable helper functions for feature construction and boundary checks.
- **Modularized ML Curator**: Split ML curator heuristics into dedicated
  analysis modules for better maintainability.
- **Optimized UI Components**: Refactored the Streamlit dashboard and
  interactive curator to use smaller, more manageable rendering and interaction
  functions.
- **Code Modernization**: Replaced redundant casts and explicit conversions
  with idiomatic Python patterns across the codebase.

### Fixed

- **Improved Error Handling**: Tightened control flow and improved safe file
  parsing in MATLAB import and export modules.
- **UI Logic Fixes**: Simplified DataFrame construction and filtering in the
  dashboard to prevent potential edge-case failures.
- **Minor Bug Fixes**: Addressed small logic issues in energy calculation and
  control flow initialization.

### Notes

- Exact MATLAB artifact parity is still in progress; use the maintained method
  plan and proof findings docs for status instead of inferring completion from
  historical implementation work.
- Older parity experiments, planned CLI ideas, and stale module-path notes are
  intentionally omitted from this maintained changelog to avoid red herrings.

## [0.1.1] - 2026-04-17

Recent work landed on 2026-04-17 and focused on a major refactoring of the
core pipeline, analysis curators, and UI components for better modularity and
maintainability.

### Added

- **Expanded Network Metrics**: `slavv analyze` and the internal network graph
  components now compute additional metrics including total length, volume,
  surface area, densities, and edge energy statistics.
- **Robust Training Loaders**: Added a training payload loader in
  `ml_curator_training.py` with better handling of mismatched arrays and
  different file types.

### Changed

- **Core Pipeline Modularization**: Refactored the main `process_image` and
  `calculate_energy_field` functions into smaller, focused helpers for progress
  emission, context initialization, and chunked calculation.
- **MATLAB Bridge Refactor**: Decomposed `import_matlab_batch` into
  stage-specific helpers (`_import_energy_stage`, `_import_vertices_stage`,
  and related routines) to improve reliability.
- **Curator Logic Extraction**: Factored inline checks and feature
  construction blocks from the curator modules into reusable helper functions.
- **UI Dashboard Decomposition**: Split the Streamlit dashboard and
  interactive curator rendering into smaller, testable functions.
- **Idiomatic Python Refactoring**: Replaced explicit float/int casts and
  redundant conversions with f-strings, comprehensions, and similar patterns
  across the project.
- **Heuristic Splitting**: Moved ML curator heuristics into dedicated analysis
  modules.

### Fixed

- **MATLAB Import Reliability**: Unified file discovery and safe loading in the
  MATLAB import surface.
- **Dashboard Data Handling**: Simplified DataFrame and row construction in the
  web dashboard to use clearer control flow and avoid unnecessary indexing.
