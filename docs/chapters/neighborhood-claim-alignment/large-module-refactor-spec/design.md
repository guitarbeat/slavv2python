# Design Document: Large Module Refactor

## Overview

This design breaks the oversized `source/slavv/` modules into smaller internal
units while preserving the current package surface and behavior.

The design is intentionally conservative:

- no algorithm rewrites
- no new product features
- no entrypoint changes
- no mandatory call-site churn up front

Instead, large files become thin orchestration surfaces that delegate to
smaller, responsibility-focused siblings.

## Design Principles

- Preserve behavior before improving elegance.
- Extract cohesive blocks, not random helpers.
- Prefer package-local modules over cross-cutting dumping grounds.
- Keep tests close to the owning subsystem.
- Use compatibility imports to avoid large caller migrations.
- Sequence the work so parity-critical paths are continuously provable.

## Priority Waves

### Wave 1: Lowest-Risk Proof Of Workflow

- `source/slavv/apps/web_app.py`
- `source/slavv/visualization/network_plots.py`
- `source/slavv/analysis/ml_curator.py`

These files are large and worth splitting, but they are not the active
imported-MATLAB parity blockers described in `TODO.md`. They are the safest
places to prove the extraction workflow before touching parity-critical code.

### Wave 2: Medium-Risk Structural Cleanup

- `source/slavv/analysis/geometry.py`
- `source/slavv/parity/reporting.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/parity/run_layout.py`

These files are operationally important, but they are not the primary live
algorithmic blocker surfaces. They should be refactored in narrow, reversible
slices after Wave 1 establishes the pattern.

### Wave 3: Deferred Parity-Sensitive Modules

- `source/slavv/core/energy.py`
- `source/slavv/core/vertices.py`
- `source/slavv/parity/metrics.py`
- `source/slavv/parity/comparison.py`
- `source/slavv/core/edge_selection.py`
- `source/slavv/core/edge_candidates.py`

These files sit close to the current parity convergence loop and should be
deferred until the active chapter is calmer, or handled only through
exceptionally small helper extractions.

## Stability Surfaces

The following surfaces must remain stable throughout the refactor:

- `pyproject.toml` console entrypoints: `slavv` and `slavv-app`
- package re-export surfaces in:
  - `source/slavv/core/__init__.py`
  - `source/slavv/analysis/__init__.py`
  - `source/slavv/runtime/__init__.py`
  - `source/slavv/parity/__init__.py`
  - `source/slavv/visualization/__init__.py`
- direct module imports already used by tests and documented workflows
- CLI and Streamlit launch paths documented in `AGENTS.md` and `README.md`

Design consequence:

- extracted helpers should move into sibling modules while the current
  top-level modules remain stable import facades until a later compatibility
  cleanup pass

## Target Module Map

### 1. `source/slavv/core/edge_candidates.py`

Current file likely mixes frontier tracing, MATLAB parity candidate
supplementation, connection resolution, diagnostics, and candidate manifest
assembly.

Target shape:

- `source/slavv/core/edge_candidates.py`
  - thin compatibility/orchestration surface
- `source/slavv/core/edge_candidate_frontier.py`
  - frontier tracing and per-origin expansion
- `source/slavv/core/edge_candidate_resolution.py`
  - parent/child ownership and terminal resolution helpers
- `source/slavv/core/edge_candidate_supplement.py`
  - watershed/geodesic/parity candidate supplementation
- `source/slavv/core/edge_candidate_diagnostics.py`
  - lifecycle records, audit summaries, diagnostics helpers

Boundary rule:

- logic that mutates candidate discovery stays separate from pure diagnostics
  summarization

### 2. `source/slavv/parity/comparison.py`

Current file likely combines CLI-facing orchestration, artifact discovery,
Python/MATLAB run execution flow, report assembly, and reuse logic.

Target shape:

- `source/slavv/parity/comparison.py`
  - top-level orchestration entrypoints
- `source/slavv/parity/comparison_runtime.py`
  - run orchestration and execution sequencing
- `source/slavv/parity/comparison_artifacts.py`
  - artifact discovery, loading, and normalized source selection
- `source/slavv/parity/comparison_manifest.py`
  - manifest/report payload assembly
- `source/slavv/parity/comparison_reuse.py`
  - reuse decisions and resume/replay helpers

Boundary rule:

- orchestration should depend on helpers, not the other way around

### 3. `source/slavv/apps/web_app.py`

Current file likely mixes Streamlit page wiring, UI state, callbacks, data
transform helpers, and rendering sections.

Target shape:

- `source/slavv/apps/web_app.py`
  - entrypoint and top-level page composition
- `source/slavv/apps/web_app_state.py`
  - session-state defaults and state transition helpers
- `source/slavv/apps/web_app_sections.py`
  - page-section renderers
- `source/slavv/apps/web_app_actions.py`
  - button-triggered actions and pipeline operations
- `source/slavv/apps/web_app_formatting.py`
  - display-only formatting and table/report helpers

Boundary rule:

- rendering helpers should not directly own pipeline mutation when that can be
  routed through action helpers

### 4. `source/slavv/visualization/network_plots.py`

Current file likely combines plotting orchestration, trace construction,
coloring, legends, and 2D/3D special cases.

Target shape:

- `source/slavv/visualization/network_plots.py`
  - public plotting entrypoints
- `source/slavv/visualization/network_plot_traces.py`
  - trace construction for vertices/edges/strands
- `source/slavv/visualization/network_plot_coloring.py`
  - coloring, opacity, and colorbar helpers
- `source/slavv/visualization/network_plot_layout.py`
  - figure/layout/aspect helpers

Boundary rule:

- trace geometry and display styling should remain separable

### 5. `source/slavv/parity/metrics.py`

Target shape:

- `source/slavv/parity/metrics.py`
  - report-level metric entrypoints
- `source/slavv/parity/metrics_vertices.py`
  - vertex matching and vertex report helpers
- `source/slavv/parity/metrics_edges.py`
  - edge pair/count/coverage metrics
- `source/slavv/parity/metrics_network.py`
  - strand/network aggregate metrics

### 6. `source/slavv/core/energy.py`

Target shape:

- `source/slavv/core/energy.py`
  - backend selection and public orchestration
- `source/slavv/core/energy_backends.py`
  - backend-specific implementations and adapters
- `source/slavv/core/energy_storage.py`
  - resumable storage and serialization helpers
- `source/slavv/core/energy_scales.py`
  - scale selection and shared energy-shape helpers

### 7. `source/slavv/analysis/ml_curator.py`

Target shape:

- `source/slavv/analysis/ml_curator.py`
  - public curator API
- `source/slavv/analysis/ml_curator_features.py`
  - feature extraction and dataset shaping
- `source/slavv/analysis/ml_curator_training.py`
  - training and model fitting
- `source/slavv/analysis/ml_curator_io.py`
  - save/load, upload handling, and model persistence safeguards

### 8. `source/slavv/runtime/run_state.py`

Target shape:

- `source/slavv/runtime/run_state.py`
  - public run-state orchestration and context API
- `source/slavv/runtime/run_state_paths.py`
  - filesystem layout and path helpers
- `source/slavv/runtime/run_state_snapshot.py`
  - snapshot read/write and schema helpers
- `source/slavv/runtime/run_state_status.py`
  - status-line rendering and summary formatting

### 9. `source/slavv/parity/run_layout.py`

Target shape:

- `source/slavv/parity/run_layout.py`
  - public layout helpers
- `source/slavv/parity/run_layout_paths.py`
  - path normalization and staged-layout resolution
- `source/slavv/parity/run_layout_inventory.py`
  - inventory/listing/index helpers
- `source/slavv/parity/run_layout_summary.py`
  - summary and manifest formatting helpers

### 10. Remaining Medium-Size Modules

Proposed conservative splits:

- `source/slavv/core/vertices.py`
  - extraction helpers
  - center-image painting / MATLAB-compat helpers
- `source/slavv/analysis/geometry.py`
  - path resampling
  - registration/transforms
  - path metrics support helpers
- `source/slavv/core/edge_selection.py`
  - cleanup policies
  - duplicate/conflict resolution
  - workflow-specific selection entrypoints
- `source/slavv/parity/reporting.py`
  - summary text builders
  - markdown/report serialization helpers

## Migration Strategy

For each oversized module:

1. Identify one cohesive extraction seam.
2. Move private helpers first.
3. Keep the original module as the import-stable facade.
4. Run targeted tests immediately after the extraction.
5. Stop and stabilize before the next seam.

This means a single source file may remain above the desired size for part of
the migration while helpers move out incrementally.

For parity-critical modules, the migration strategy is stricter:

- extract diagnostics or formatting helpers before mutation logic
- avoid moving algorithmically sensitive decision code unless the slice is very
  small and backed by the nearest parity regression tests
- prefer deferral over a broad cleanup if the active chapter is using the file
  as a live debugging surface

## Compatibility Strategy

- Public entrypoints stay in their current modules initially.
- Existing tests may continue importing the original modules.
- New sibling modules host extracted helpers, typically private first.
- Re-export only what current callers need.
- Remove temporary shims only after a later cleanup pass with dedicated proof.

## Stability Surfaces That Must Remain Intact

### Packaging And Entrypoints

The following package entrypoints are explicitly stable and should not move:

- `slavv = "slavv.apps.cli:main"` in [pyproject.toml](C:/Users/alw4834/Documents/slavv2python/pyproject.toml)
- `slavv-app = "slavv.apps.streamlit_launcher:main"` in [pyproject.toml](C:/Users/alw4834/Documents/slavv2python/pyproject.toml)

Implication:

- refactoring `source/slavv/apps/web_app.py` must not alter the launcher
  contract documented in [README.md](C:/Users/alw4834/Documents/slavv2python/README.md)
- refactoring parity modules must not break `source/slavv/apps/parity_cli.py`
  imports from `slavv.parity.comparison` and `slavv.parity.run_layout`

### Package Re-Exports

The following import surfaces are already consumed by tests and should stay
stable during extraction:

- `from slavv.core import SLAVVProcessor` via
  [source/slavv/core/__init__.py](C:/Users/alw4834/Documents/slavv2python/source/slavv/core/__init__.py)
- `from slavv.analysis import ...` geometry and ML curator exports via
  [source/slavv/analysis/__init__.py](C:/Users/alw4834/Documents/slavv2python/source/slavv/analysis/__init__.py)
- `from slavv.runtime import RunContext, load_run_snapshot, ...` via
  [source/slavv/runtime/__init__.py](C:/Users/alw4834/Documents/slavv2python/source/slavv/runtime/__init__.py)
- lazy parity exports such as `orchestrate_comparison`, `compare_edges`,
  `generate_summary`, and `load_parameters` via
  [source/slavv/parity/__init__.py](C:/Users/alw4834/Documents/slavv2python/source/slavv/parity/__init__.py)
- `from slavv.visualization import NetworkVisualizer` via
  [source/slavv/visualization/__init__.py](C:/Users/alw4834/Documents/slavv2python/source/slavv/visualization/__init__.py)

Implication:

- helper extraction should prefer moving private functions first
- if a symbol currently re-exported from a package `__init__` moves, the
  original export path must continue working

### Direct Module Imports Used By Tests

Some large modules are imported directly by tests and therefore need stronger
import compatibility than package-only surfaces:

- `slavv.core.edge_candidates`
- `slavv.core.energy`
- `slavv.core.edge_selection`
- `slavv.core.vertices`
- `slavv.parity.comparison`
- `slavv.parity.metrics`
- `slavv.parity.run_layout`
- `slavv.parity.reporting`
- `slavv.apps.web_app`
- `slavv.visualization.network_plots`

Implication:

- these modules should remain as top-level facades even after helpers move out

### Documentation-Coupled Workflow Surfaces

The following documented user workflows should remain unchanged during the
refactor:

- `slavv info`, `slavv run`, `slavv analyze`, `slavv plot`, and
  `slavv import-matlab` in
  [README.md](C:/Users/alw4834/Documents/slavv2python/README.md)
- `slavv-app` and `python -m streamlit run source/slavv/apps/web_app.py` in
  [README.md](C:/Users/alw4834/Documents/slavv2python/README.md)
- extraction extension guidance that explicitly references `source/slavv/apps/cli.py`
  in [docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md](C:/Users/alw4834/Documents/slavv2python/docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md)

Implication:

- the refactor should not rename or relocate documented entry modules without a
  deliberate docs update and compatibility plan

## Parity-Critical Modules

Current chapter and TODO context make these modules especially sensitive:

- `source/slavv/core/edge_candidates.py`
- `source/slavv/parity/comparison.py`
- `source/slavv/parity/metrics.py`
- `source/slavv/parity/run_layout.py`
- `source/slavv/parity/reporting.py`
- `source/slavv/runtime/run_state.py`

These files should still be refactored, but only in narrow slices with focused
regression proof. `edge_candidates.py` remains the highest-value target, but it
should be approached as multiple small extractions rather than one large move.

## Verification Strategy

Per slice:

- run targeted `pytest` for the owning subsystem
- run `python -m ruff check source dev/tests`
- run `python -m mypy` when the touched files are in the mypy-covered surface

Per wave:

- run `python -m pytest -m "unit or integration"`

For parity-critical extractions:

- include the most relevant parity/unit diagnostic tests
- if the slice touches run-layout or runtime surfaces, include the related
  diagnostics and runtime tests listed in `AGENTS.md`

## Risks And Mitigations

- Risk: helper extraction changes import order or circular dependencies.
  - Mitigation: keep orchestration facades thin and extract leaf helpers first.
- Risk: parity logic drifts during structural edits.
  - Mitigation: avoid semantic rewrites and run targeted regression tests after
    every parity-related slice.
- Risk: web app refactor breaks session-state assumptions.
  - Mitigation: isolate state initialization first and preserve existing keys.
- Risk: plotting refactors change figure defaults subtly.
  - Mitigation: keep public plotting entrypoints stable and rely on current UI
    tests before broad cleanup.

## Exit Criteria

The refactor program is complete when:

- each oversized module has either been split or explicitly deferred with a
  documented reason
- the remaining top-level modules act mainly as stable entry surfaces
- tests and docs reflect the new ownership boundaries
- no compatibility shim remains without a tracked cleanup task
