# Changelog

This file summarizes notable repository changes for the maintained SLAVV Python
codebase.

The repository does not currently use tags or formal release notes, so this
file stays intentionally lightweight. It highlights the current product surface
and recent changes without trying to preserve superseded workflow plans as if
they were still active.

For current behavior and proof status, prefer:

- [README.md](../README.md)
- [AGENTS.md](../AGENTS.md)
- [MATLAB Method Implementation Plan](reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [Exact Proof Findings](reference/core/EXACT_PROOF_FINDINGS.md)

## [Unreleased] - 2026-06-24

### Added

- **Random Component Parity Suite** (ADR 0010): Seeded white-noise MATLAB R2019a/Python differential loop for fast Energy building-block checks (linspace, `interp3`, padded shape, valid flags). Structural fields gate CI; Hessian float ULP is advisory only. Includes the accompanying package refactor documented in [ADR 0010](adr/0010-random-component-parity-suite.md).
- **High-level Python SLAVV facade** (`slavv_python/pipeline/slavv_vectorize.py`): `vectorize_python(image, params)` orchestrator equivalent to `vectorize_V200.m`, plus thin `get_*_python` convenience wrappers over the exact-parity stage managers.
- **ADR 0011 (Proposed)** — Energy float certification policy: strict scale winners vs bounded float64 ULP tolerance on `energy.energy` ([adr/0011-energy-float-certification-policy.md](adr/0011-energy-float-certification-policy.md)).
- **Crop Energy oracle v2**: Refreshed MATLAB oracle vectors for the `180709_E` crop Energy proof.

## [Unreleased] - 2026-06-09

### Added

- **Automated Parity Job Monitoring System**: Track long-running parity experiments with automatic notifications
  - **JobRegistry** (`analytics/parity/job_registry.py`): Persistent JSONL-based job tracking with file locking for concurrent access protection
  - **MonitorDaemon** (`analytics/parity/monitor_daemon.py`): Background daemon monitoring jobs every 30 seconds, sends desktop notifications on completion/failure, auto-terminates after 60 minutes of idle time
  - **Process utilities** (`analytics/parity/process_utils.py`): Process liveness checking with `psutil`, PID reuse protection via process name validation, process tree termination
  - **`slavv jobs` CLI commands**: 
    - `slavv jobs list` - View all active monitored jobs in table format
    - `slavv jobs history [--run-dir PATH] [--limit N]` - Query completed jobs with optional filters
    - `slavv jobs kill <job-id>` - Terminate running jobs by ID (accepts partial IDs)
    - `slavv jobs daemon status` - Check daemon PID, uptime, and active job count
    - `slavv jobs daemon restart` - Restart the monitoring daemon
  - **`--monitor` flag** for `resume-exact-run` and `launch-exact-run`: Enable automatic job tracking and notifications for long-running experiments
  - **`--force-kill` flag**: Terminate active writers before starting new jobs on the same run directory
  - **Duplicate writer prevention**: Automatic detection of concurrent writes to same run directory with clear error messages and remediation options
  - **Desktop notifications (Windows)**: Toast notifications via `win10toast` library with graceful fallback to logging when unavailable
  - **Job history persistence**: Survives terminal restarts, IDE restarts, and system reboots via JSONL append-only storage
  - **Cross-session monitoring**: Daemon persists across terminal closures and can monitor jobs started in different sessions
  - **Comprehensive documentation**: [PARITY_JOB_MONITORING.md](reference/workflow/PARITY_JOB_MONITORING.md) with architecture overview, usage examples, troubleshooting guide, and best practices

### Changed

- **Parity experiment CLI**: `resume-exact-run` and `launch-exact-run` now support `--monitor` and `--force-kill` flags
- **Dependencies**: Added `fasteners>=0.18.0` (file locking), `tabulate>=0.9.0` (CLI tables), `win10toast>=0.9` (Windows notifications to workspace extras)
- **Documentation updates**:
  - [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md): Added monitoring examples and `slavv jobs` commands for Tier 2 crop harness workflow
  - [PARITY_CERTIFICATION_GUIDE.md](reference/workflow/PARITY_CERTIFICATION_GUIDE.md): Updated with monitoring best practices for long-running certification runs
  - [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md): Updated cold-start protocol to include `slavv jobs list` check and `--monitor` flag recommendations

## [Unreleased] - 2026-05-27

### Added

- **EnergyManager** (`energy/manager.py`): Ephemeral `run()` and resumable `run_resumable()` share one Energy Field pipeline; orchestrator energy stage uses manager directly.
- **ExactProofCoordinator** (`analytics/parity/coordinator.py`): Unified `prove`, `capture_candidates`, and run-count normalization; candidate capture routes through `EdgeManager.discover_candidates()`.
- **VertexManager** (`vertices/manager.py`): Ephemeral `run()` and resumable `run_resumable()` share one Vertex Set pipeline; vertex detection moved from `edges/candidate_detection.py` to `vertices/detection.py`.
- **ADR 0007** (VertexManager) and **ADR 0008** (ExactProofCoordinator + `counts.py` seam).
- **Edge discovery strategy seam** (`discovery.py`): `CandidateManifest`, `MaintainedTracingDiscovery`, `FrontierTracingDiscovery`, and `select_edge_discovery()` for tracing vs MATLAB-parity frontier branching.
- **NetworkManager** (`network/manager.py`): Ephemeral `run()` and resumable `run_resumable()` share one graph-build pipeline; **ADR 0006** documents the lifecycle manager.
- **Run ledger modules** (`engine/state/run_ledger.py`, `engine/state/stage_handle.py`): `RunContext` and `StageController` implementations moved out of `engine/context.py` (thin re-export barrel).
- **AppRunState** (`schema/app_run.py`): Typed UI envelope holding `PipelineResult`; session stores `AppRunState` until export boundaries call `.to_dict()`.
- **ADR 0005**: Documents the edge discovery strategy seam; ADR 0003 updated with ephemeral `run()` completion.

### Changed

- **Global watershed strel seed selection**: Tied adjusted energies inside the structuring element now break on lowest Fortran linear index (MATLAB parity), via `_argmin_with_linear_index_tiebreak` in `edges/matlab_indexing.py`.
- **Parity CLI**: `handle_prove_exact` / `handle_capture_candidates` construct `ExactProofCoordinator` explicitly.
- **Energy extraction**: `calculate_energy_field` / `calculate_energy_field_resumable` delegate to `EnergyManager`.
- **EdgeManager unify**: `EdgeManager.run()` and `run_resumable()` share `_run_tracing()`; removed duplicate `extraction_standard.py`. Orchestrator and `extract_edges()` use `EdgeManager.run()` for ephemeral runs.
- **EdgeManager consolidation**: `run_resumable()` is the single resumable tracing entrypoint (audit JSON, parity candidate checkpoints, lifecycle artifacts, selection, bridging, finalize). Removed the 14-callable `resumable.extract_edges_resumable` injection surface.
- **Vertex extraction**: `extract_vertices` / `extract_vertices_resumable` delegate to `VertexManager`; orchestrator vertices stage uses manager directly.
- **Network construction**: `construct_network` / `construct_network_resumable` delegate to `NetworkManager`; orchestrator network stage uses manager directly.
- **Typed pipeline output**: `SlavvPipeline.run()` returns `PipelineResult` (`Mapping`-compatible for legacy `results["key"]` access). `StageExecutor` persists checkpoints via schema `.save()` / `.load()` when available.
- **Interface shared state**: `store_processing_session_state` and curation helpers prefer `AppRunState` / `PipelineResult` over immediate dict normalization.
- **Documentation**: Updated technical architecture, naming guide, MATLAB parity mapping, GEMINI repo map, and ADRs 0001–0006 (including `AppRunState` in ADR 0001 and PYTHON_NAMING_GUIDE).

### Removed

- **`extraction_standard.py`**: Ephemeral edge tracing now routes through `EdgeManager.run()`.

## [Unreleased] - 2026-05-22

### Added

- **CI Paper Profile Integration Test**: Implemented `test_paper_profile_ci.py` using synthetic TIFF data to ensure the paper profile pipeline works in CI/CD without real dataset dependencies.
- **CI Regression Gate Update**: Updated GitHub Actions workflow to run the new synthetic integration test along with unit tests.

### Fixed

- **Parity: Global Watershed Alignment**: Implemented "Measure 3" tightening by adding a hard accumulated distance ($d/R > 3.0$) expansion cutoff, matching MATLAB's `get_edges_by_watershed` behavior.
- **Parity: Edge Influence Sigma**: Updated default `sigma_per_influence_edges` to $2/3$, aligning with MATLAB's conflict painting regions.
- **Parity: Global Watershed Tie-Breaking**: Replaced `np.isclose` with bit-exact equality and added linear index priority to the frontier priority queue, matching MATLAB's hub vertex exploration behavior.
- **Parity: Energy Precision**: Removed all remaining `float32` casts in watershed suppression, tolerance checks, and trace sampling, enforcing `float64` bit-accuracy across the expansion frontier.
- **Parity: Edge Budget**: Removed the incorrect `edge_number_tolerance` override for the exact route, ensuring high-degree hubs can initiate 4 exploratory traces to match the MATLAB oracle.
- **Test Suite: Object Attribute Access**: Resolved 20+ unit test failures by updating tests to use attributes (`.energy`, `.traces`, etc.) instead of subscripting (`["energy"]`) on the new typed `EnergyResult`, `EdgeSet`, `VertexSet`, and `NetworkResult` objects.
- **Resumable Edge Extraction**: Fixed bugs in `resumable_edges.py` where `EnergyResult` and `EdgeSet` objects were being subscripted or incorrectly assigned, resolving critical failures when running with checkpoints.
- **Network Construction Result**: Fixed a `TypeError` in `construct_network` where it attempted item assignment on a typed result object.
- **Edge Candidate Seeding**: Fixed `IndexError` in edge tracing when `scale_indices` were missing from the energy result payload.

## [Unreleased] - 2026-05-15

### Fixed

- **MATLAB Parity Fix: Parameter Realignment**: Discovered and corrected a mismatch in `edge_number_tolerance` (2 -> 4), resolving missing connections for high-degree junction vertices.
- **MATLAB Parity Fix: NaN Stability**: Fixed a critical bug where multiplying `-Inf` vertex priority by `0.0` suppression factors produced `NaNs`, corrupting the frontier seed selection order.
- **Reliability: Python 3.7 Compatibility**: Fixed `ImportError` by migrating `typing.Protocol` to `typing_extensions.Protocol`.

## [Unreleased] - 2026-05-14

### Fixed

- **MATLAB Parity Fix: Vertex Priority**: Implemented priority selection for all vertices by initializing them to `-Inf` in the frontier map, ensuring they are processed as sources before exploratory traces leapfrog them.
- **MATLAB Parity Fix: Frontier Splice Logic**: Corrected the off-by-one error in available-location insertion that caused drift from MATLAB's deterministic crawler ordering.
- **MATLAB Parity Fix: Loop Break Condition**: Aligned the watershed loop termination to break immediately upon encountering non-negative energies, matching MATLAB early-exit semantics.
- **MATLAB Parity Fix: Energy Tolerance Multiplier**: Verified and standardized the multiplier formula for energy thresholds (`1 - energy_tolerance`).

## [Unreleased] - 2026-05-05

### Added

- **Comprehensive Watershed Test Suite**: Expanded unit tests for the global watershed algorithm from 25 to 54 passing tests, covering frontier ordering, join cleanup, and penalty math.
- **Normalized Distance Scaling**: Implemented correct `r/R` scaling for watershed energy penalties, matching MATLAB's relative penalty behavior across different vessel scales.

### Changed

- **Refactored Global Watershed Module**: Massively decomposed the 800+ line watershed orchestration function into modular, well-typed helpers for state initialization, size map preparation, and result assembly.
- **Enhanced Type Safety**: Applied specialized numpy array type aliases (`Float32Array`, `Int64Array`, etc.) throughout the watershed and frontier modules to improve static analysis and readability.
- **Improved Energy Map Integrity**: Refactored watershed discovery logic to stop propagating penalized energies back to the shared energy map, ensuring frontier sorting always uses original unpenalized energies as intended.

### Fixed

- **MATLAB Parity Fix: Distance Normalization**: Resolved a significant divergence where Python used absolute micron distances for penalties instead of normalized `r/R` ratios.
- **MATLAB Parity Fix: Directional Suppression**: Verified and corrected the placement of directional suppression; confirmed that iterative suppression inside the seed loop matches MATLAB's ground truth.
- **MATLAB Parity Fix: Trace Order RNG**: Verified the fix for deterministic trace order randomization in conflict painting using seeded RNG.
- **Test Suite: Numpy Boolean Comparison**: Fixed a critical test suite failure caused by incorrect identity comparison (`is`) for numpy boolean elements.

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
