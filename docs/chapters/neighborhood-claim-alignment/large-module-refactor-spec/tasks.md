# Implementation Plan: Large Module Refactor

## Overview

This plan executes a conservative, spec-driven refactor of the largest
`source/slavv/` modules. It is intentionally phased so that structural cleanup
does not interrupt ongoing parity work or introduce broad behavioral risk.

## Status Update: 2026-04-17

Facade/package refactors landed and validated so far:

- `source/slavv/apps/web_app.py` is now a thin facade with page logic split into:
  - `source/slavv/apps/web_app_dashboard_page.py`
  - `source/slavv/apps/web_app_processing_page.py`
  - `source/slavv/apps/web_app_curation_page.py`
  - `source/slavv/apps/web_app_visualization_page.py`
  - `source/slavv/apps/web_app_analysis_page.py`
  - `source/slavv/apps/web_app_static_pages.py`
  - `source/slavv/apps/web_app_shell.py`
- `source/slavv/visualization/network_plots.py` is now a thin facade over:
  - `source/slavv/visualization/network_plot_spatial_2d.py`
  - `source/slavv/visualization/network_plot_spatial_3d.py`
  - `source/slavv/visualization/network_plot_statistics.py`
  - plus the existing helper modules under `source/slavv/visualization/`
- `source/slavv/core/edge_candidates.py` is now a thin facade over the private
  package `source/slavv/core/_edge_candidates/`
- `source/slavv/apps/cli.py` is now a thin facade over:
  - `source/slavv/apps/cli_parser.py`
  - `source/slavv/apps/cli_shared.py`
  - `source/slavv/apps/cli_exported_network.py`
  - `source/slavv/apps/cli_commands.py`
- `source/slavv/parity/comparison.py` now delegates artifact/report,
  Python-source, analysis, MATLAB-runner, reuse, health-check, and standalone
  workflow logic into `source/slavv/parity/_comparison/`
- `source/slavv/parity/metrics.py` now has an initial private-package split
  under `source/slavv/parity/_metrics/`
- `source/slavv/analysis/ml_curator.py` has been reduced by moving heuristic
  curators and support helpers into sibling modules

Current `source/` files still above 500 lines:

- `source/slavv/parity/comparison.py` — `1120`
- `source/slavv/parity/metrics.py` — `1054`
- `source/slavv/core/energy.py` — `1030`
- `source/slavv/runtime/run_state.py` — `904`
- `source/slavv/parity/run_layout.py` — `836`
- `source/slavv/core/vertices.py` — `697`
- `source/slavv/analysis/geometry.py` — `677`
- `source/slavv/core/edge_selection.py` — `653`
- `source/slavv/parity/workflow_assessment.py` — `630`
- `source/slavv/parity/reporting.py` — `613`
- `source/slavv/analysis/ml_curator.py` — `579`
- `source/slavv/core/edge_primitives.py` — `559`
- `source/slavv/core/edges.py` — `554`

## Phase 1: Inventory And Guardrails

- [x] 1. Confirm the refactor file inventory
  - [x] 1.1 Re-run the line-count inventory for `source/slavv/`
  - [x] 1.2 Freeze the initial target list of modules above 600 lines
  - [x] 1.3 Identify parity-critical modules that need narrower slice plans instead of broad extraction

- [x] 2. Define validation gates
  - [x] 2.1 Map each target module to its nearest test files in `dev/tests/`
  - [x] 2.2 Define the minimum verification command set for each subsystem
  - [x] 2.3 Mark which targets sit inside the current mypy-covered surface

### Phase 1 Findings

Frozen large-module target list:

- `source/slavv/apps/web_app.py`
- `source/slavv/core/edge_candidates.py`
- `source/slavv/parity/comparison.py`
- `source/slavv/visualization/network_plots.py`
- `source/slavv/parity/metrics.py`
- `source/slavv/core/energy.py`
- `source/slavv/analysis/ml_curator.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/parity/run_layout.py`
- `source/slavv/core/vertices.py`
- `source/slavv/analysis/geometry.py`
- `source/slavv/core/edge_selection.py`
- `source/slavv/parity/reporting.py`

Parity-critical targets that need narrow extraction slices:

- `source/slavv/core/edge_candidates.py`
- `source/slavv/parity/comparison.py`
- `source/slavv/parity/metrics.py`
- `source/slavv/parity/run_layout.py`
- `source/slavv/parity/reporting.py`
- `source/slavv/runtime/run_state.py`

Targets inside the current mypy-covered surface:

- `source/slavv/apps/web_app.py`
- `source/slavv/core/edge_candidates.py`
- `source/slavv/core/energy.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/core/vertices.py`
- `source/slavv/analysis/geometry.py`
- `source/slavv/core/edge_selection.py`

### Validation Map

- `source/slavv/core/edge_candidates.py`
  - tests: `dev/tests/unit/core/test_edge_cases.py`, `dev/tests/unit/core/test_frontier_tracing.py`, `dev/tests/unit/core/test_candidate_diagnostics.py`, `dev/tests/unit/core/test_reachability_bottleneck.py`, `dev/tests/unit/core/test_watershed_supplement_regression.py`, `dev/tests/integration/test_regression_edges.py`
  - commands: `python -m pytest dev/tests/unit/core/test_edge_cases.py dev/tests/unit/core/test_frontier_tracing.py dev/tests/unit/core/test_candidate_diagnostics.py`
  - commands: `python -m pytest dev/tests/unit/core/test_reachability_bottleneck.py dev/tests/unit/core/test_watershed_supplement_regression.py dev/tests/integration/test_regression_edges.py`

- `source/slavv/parity/comparison.py`
  - tests: `dev/tests/unit/parity/test_comparison_runtime.py`, `dev/tests/unit/parity/test_comparison_quick_view.py`, `dev/tests/unit/parity/test_source_selection.py`, `dev/tests/integration/parity/test_diagnostics_and_proof_workflow.py`
  - commands: `python -m pytest dev/tests/unit/parity/test_comparison_runtime.py dev/tests/unit/parity/test_source_selection.py`
  - commands: `python -m pytest dev/tests/unit/parity/test_comparison_quick_view.py dev/tests/integration/parity/test_diagnostics_and_proof_workflow.py`

- `source/slavv/apps/web_app.py`
  - tests: `dev/tests/ui/test_app_integration.py`, `dev/tests/ui/test_app_layout.py`, `dev/tests/unit/apps/test_web_app_dashboard.py`, `dev/tests/unit/apps/test_streamlit_launcher.py`, `dev/tests/ui/test_share_report_entrypoint.py`
  - commands: `python -m pytest dev/tests/unit/apps/test_web_app_dashboard.py dev/tests/unit/apps/test_streamlit_launcher.py`
  - commands: `python -m pytest dev/tests/ui/test_app_integration.py dev/tests/ui/test_app_layout.py dev/tests/ui/test_share_report_entrypoint.py`

- `source/slavv/visualization/network_plots.py`
  - tests: `dev/tests/ui/test_edge_coloring.py`, `dev/tests/ui/test_visualization_aspect.py`, `dev/tests/ui/test_network_slice.py`, `dev/tests/ui/test_flow_field.py`, `dev/tests/ui/test_animate_strands.py`, `dev/tests/ui/test_visualization_exports.py`, `dev/tests/unit/io/test_casx_export.py`, `dev/tests/unit/io/test_casx_export_full.py`, `dev/tests/unit/io/test_mat_io.py`, `dev/tests/unit/io/test_vmv_export.py`
  - commands: `python -m pytest dev/tests/ui/test_edge_coloring.py dev/tests/ui/test_visualization_aspect.py dev/tests/ui/test_network_slice.py dev/tests/ui/test_flow_field.py dev/tests/ui/test_animate_strands.py dev/tests/ui/test_visualization_exports.py`
  - commands: `python -m pytest dev/tests/unit/io/test_casx_export.py dev/tests/unit/io/test_casx_export_full.py dev/tests/unit/io/test_mat_io.py dev/tests/unit/io/test_vmv_export.py`

- `source/slavv/parity/metrics.py`
  - tests: `dev/tests/unit/parity/test_comparison_metrics.py`, `dev/tests/unit/parity/test_comparison_quick_view.py`
  - commands: `python -m pytest dev/tests/unit/parity/test_comparison_metrics.py dev/tests/unit/parity/test_comparison_quick_view.py`

- `source/slavv/core/energy.py`
  - tests: `dev/tests/unit/core/test_energy_methods.py`, `dev/tests/unit/core/test_energy_field_storage.py`, `dev/tests/unit/core/test_direction_method.py`, `dev/tests/unit/core/test_discrete_tracing.py`, `dev/tests/unit/core/test_progress_callback.py`, `dev/tests/integration/test_public_api.py`, `dev/tests/integration/test_regression_edges.py`
  - commands: `python -m pytest dev/tests/unit/core/test_energy_methods.py dev/tests/unit/core/test_energy_field_storage.py dev/tests/unit/core/test_direction_method.py dev/tests/unit/core/test_discrete_tracing.py dev/tests/unit/core/test_progress_callback.py`
  - commands: `python -m pytest dev/tests/integration/test_public_api.py dev/tests/integration/test_regression_edges.py`

- `source/slavv/analysis/ml_curator.py`
  - tests: `dev/tests/unit/analysis/test_choose_heuristics.py`, `dev/tests/unit/analysis/test_feature_alignment.py`, `dev/tests/unit/analysis/test_ml_curator_improvements.py`, `dev/tests/unit/analysis/test_ml_curator_security.py`, `dev/tests/unit/analysis/test_ml_model_io.py`, `dev/tests/unit/analysis/test_ml_training.py`, `dev/tests/integration/test_automatic_curator.py`, `dev/tests/integration/test_uncurated_info.py`
  - commands: `python -m pytest dev/tests/unit/analysis/test_choose_heuristics.py dev/tests/unit/analysis/test_feature_alignment.py dev/tests/unit/analysis/test_ml_curator_improvements.py dev/tests/unit/analysis/test_ml_curator_security.py dev/tests/unit/analysis/test_ml_model_io.py dev/tests/unit/analysis/test_ml_training.py`
  - commands: `python -m pytest dev/tests/integration/test_automatic_curator.py dev/tests/integration/test_uncurated_info.py`

- `source/slavv/runtime/run_state.py`
  - tests: `dev/tests/unit/runtime/test_run_state.py`, `dev/tests/unit/apps/test_web_app_dashboard.py`, `dev/tests/unit/core/test_resumable_frontier_provenance.py`, `dev/tests/integration/parity/test_diagnostics_and_proof_workflow.py`
  - commands: `python -m pytest dev/tests/unit/runtime/test_run_state.py dev/tests/unit/core/test_resumable_frontier_provenance.py dev/tests/unit/apps/test_web_app_dashboard.py`
  - commands: `python -m pytest dev/tests/integration/parity/test_diagnostics_and_proof_workflow.py`

- `source/slavv/parity/run_layout.py`
  - tests: `dev/tests/unit/parity/test_parity_layouts.py`, `dev/tests/unit/parity/test_comparison_runtime.py`, `dev/tests/unit/workspace_scripts/test_comparison_layout_smoothing.py`, `dev/tests/integration/parity/test_diagnostics_and_proof_workflow.py`
  - commands: `python -m pytest dev/tests/unit/parity/test_parity_layouts.py dev/tests/unit/parity/test_comparison_runtime.py`
  - commands: `python -m pytest dev/tests/unit/workspace_scripts/test_comparison_layout_smoothing.py dev/tests/integration/parity/test_diagnostics_and_proof_workflow.py`

- `source/slavv/core/vertices.py`
  - tests: `dev/tests/unit/core/test_edge_cases.py`, `dev/tests/unit/core/test_frontier_tracing.py`, `dev/tests/unit/core/test_network_cycles.py`, `dev/tests/unit/core/test_resumable_frontier_provenance.py`, `dev/tests/unit/core/test_watershed_edges.py`, `dev/tests/integration/test_public_api.py`, `dev/tests/integration/test_regression_edges.py`, `dev/tests/integration/test_end_to_end_pipeline.py`
  - commands: `python -m pytest dev/tests/unit/core/test_edge_cases.py dev/tests/unit/core/test_frontier_tracing.py dev/tests/unit/core/test_network_cycles.py dev/tests/unit/core/test_resumable_frontier_provenance.py dev/tests/unit/core/test_watershed_edges.py`
  - commands: `python -m pytest dev/tests/integration/test_public_api.py dev/tests/integration/test_regression_edges.py`

- `source/slavv/analysis/geometry.py`
  - tests: `dev/tests/unit/analysis/test_vector_geometry.py`, `dev/tests/unit/analysis/test_network_stats.py`, `dev/tests/unit/analysis/test_cropping.py`
  - commands: `python -m pytest dev/tests/unit/analysis/test_vector_geometry.py dev/tests/unit/analysis/test_network_stats.py dev/tests/unit/analysis/test_cropping.py`

- `source/slavv/core/edge_selection.py`
  - tests: `dev/tests/unit/core/test_edge_cases.py`, `dev/tests/unit/core/test_candidate_diagnostics.py`, `dev/tests/integration/test_regression_edges.py`
  - commands: `python -m pytest dev/tests/unit/core/test_edge_cases.py dev/tests/unit/core/test_candidate_diagnostics.py`
  - commands: `python -m pytest dev/tests/integration/test_regression_edges.py`

- `source/slavv/parity/reporting.py`
  - tests: `dev/tests/unit/parity/test_parity_layouts.py`, `dev/tests/unit/parity/test_comparison_quick_view.py`
  - commands: `python -m pytest dev/tests/unit/parity/test_parity_layouts.py dev/tests/unit/parity/test_comparison_quick_view.py`

### Minimum Verification By Subsystem

- `apps` and `visualization`
  - run targeted UI/unit tests
  - run `python -m ruff check source dev/tests`
  - run `python -m mypy` when `web_app.py` changes

- `core`, `runtime`, and `analysis`
  - run targeted unit/integration tests
  - run `python -m ruff check source dev/tests`
  - run `python -m mypy` for mypy-covered targets

- `parity`
  - run targeted parity unit tests plus the nearest integration proof path
  - run `python -m ruff check source dev/tests`
  - run `python -m pytest -m "unit or integration"` at the end of each parity-heavy wave

## Phase 2: Wave 1 Extraction Plan

- [x] 3. Refactor `source/slavv/apps/web_app.py`
  - [x] 3.1 Extract dashboard frame and placeholder helpers into `web_app_dashboard.py`
  - [x] 3.2 Extract export/share-report artifact helpers into `web_app_artifacts.py`
  - [x] 3.3 Extract dashboard filtering and backlog pure helpers into `web_app_dashboard.py`
  - [x] 3.4 Extract session-state initialization helpers
  - [x] 3.5 Extract action/callback helpers
  - [x] 3.6 Extract section-rendering helpers
  - [x] 3.7 Keep `web_app.py` as the top-level entrypoint
  - [x] 3.8 Run relevant app-smoke and refactor regression tests

- [x] 4. Refactor `source/slavv/visualization/network_plots.py`
  - [x] 4.1 Extract trace-construction helpers
  - [x] 4.2 Extract coloring/colorbar helpers into `network_plot_helpers.py`
  - [x] 4.3 Extract axis/layout/title helpers into `network_plot_layout.py`
  - [x] 4.4 Preserve existing plotting entrypoints
  - [x] 4.5 Run relevant visualization and UI tests

- [x] 5. Refactor `source/slavv/analysis/ml_curator.py`
  - [x] 5.1 Extract feature engineering helpers
  - [x] 5.2 Extract training-data aggregation helpers into `ml_curator_training.py`
  - [x] 5.3 Extract model I/O materialization helpers into `ml_curator_io.py`
  - [x] 5.4 Preserve current curator-facing APIs
  - [x] 5.5 Run relevant analysis and integration tests

- [x] 6. Validate Wave 1
  - [x] 6.1 Run targeted `ruff` checks for touched Wave 1 files
  - [x] 6.2 Run `python -m mypy`
  - [x] 6.3 Run focused Wave 1 pytest commands for apps, visualization, and analysis
  - [x] 6.4 Run `python -m pytest -m "unit or integration"` as a broader checkpoint
  - [x] 6.5 Record any circular-import or compatibility issues before Wave 2

### Wave 1 Progress Notes

Completed helper modules introduced so far:

- `source/slavv/apps/web_app_dashboard.py`
- `source/slavv/apps/web_app_artifacts.py`
- `source/slavv/apps/web_app_dashboard_page.py`
- `source/slavv/apps/web_app_processing_page.py`
- `source/slavv/apps/web_app_curation_page.py`
- `source/slavv/apps/web_app_visualization_page.py`
- `source/slavv/apps/web_app_analysis_page.py`
- `source/slavv/apps/web_app_static_pages.py`
- `source/slavv/apps/web_app_shell.py`
- `source/slavv/visualization/network_plot_dashboard.py`
- `source/slavv/visualization/network_plot_helpers.py`
- `source/slavv/visualization/network_plot_layout.py`
- `source/slavv/visualization/network_plot_spatial_2d.py`
- `source/slavv/visualization/network_plot_spatial_3d.py`
- `source/slavv/visualization/network_plot_statistics.py`
- `source/slavv/analysis/ml_curator_features.py`
- `source/slavv/analysis/ml_curator_io.py`
- `source/slavv/analysis/ml_curator_training.py`
- `source/slavv/apps/cli_parser.py`
- `source/slavv/apps/cli_shared.py`
- `source/slavv/apps/cli_exported_network.py`
- `source/slavv/apps/cli_commands.py`
- `source/slavv/core/_edge_candidates/`
- `source/slavv/parity/_comparison/analysis.py`
- `source/slavv/parity/_comparison/artifacts.py`
- `source/slavv/parity/_comparison/python_sources.py`
- `source/slavv/parity/_comparison/task_recording.py`
- `source/slavv/parity/_comparison/matlab_runner.py`
- `source/slavv/parity/_comparison/reuse.py`
- `source/slavv/parity/_comparison/health_check.py`
- `source/slavv/parity/_comparison/standalone.py`
- `source/slavv/parity/_metrics/`

Resolved compatibility notes so far:

- `source/slavv/visualization/network_plots.py` and `source/slavv/visualization/network_plot_layout.py` now use a consistent helper API for shared 3D scene and simple figure layouts (`plot_3d_scene_layout`, `figure_layout`) after an in-progress extraction briefly left import/call mismatches.
- `source/slavv/visualization/network_plot_layout.py` now also owns the shared empty-state figure helper used by visualization methods that previously duplicated centered "no data" annotations inline.
- `rope` was successfully used to extract the summary-dashboard trace population block in `source/slavv/visualization/network_plots.py` into a dedicated helper method, with follow-up type tightening and regression coverage added afterward.
- `source/slavv/visualization/network_plots.py` now calls helper modules directly for color mapping, colorbars, and summary-dashboard trace assembly, with the internal wrapper methods removed to produce net LOC reduction.
- `source/slavv/analysis/ml_curator.py` now imports gradient, bounds, and feature-importance helpers from `ml_curator_features.py`, removing internal helper methods while preserving the public curator surface.
- `source/slavv/apps/web_app.py` preserves import-time Streamlit page config and CSS injection while re-exporting the tested app/page/helper symbols from the new sibling modules.
- `source/slavv/apps/cli.py` preserves the tested CLI entry surface while delegating parser, shared, export, and command implementations to sibling modules.
- `source/slavv/core/edge_candidates.py` preserves the old monkeypatch/import surface while delegating implementation into `source/slavv/core/_edge_candidates/`.
- `source/slavv/parity/comparison.py` preserves monkeypatch-sensitive facade attributes such as `run_matlab_vectorization`, `load_matlab_batch_results`, `compare_results`, `generate_summary`, and `generate_manifest` while delegating implementation into `source/slavv/parity/_comparison/`.

Completed focused regression tests introduced so far:

- `dev/tests/unit/apps/test_web_app_dashboard_refactor.py`
- `dev/tests/unit/apps/test_web_app_artifacts_refactor.py`
- `dev/tests/unit/analysis/test_ml_curator_io.py`
- `dev/tests/unit/analysis/test_ml_curator_features.py`
- `dev/tests/unit/analysis/test_ml_curator_training.py`
- `dev/tests/unit/visualization/test_network_plot_layout.py`
- `dev/tests/ui/test_summary_dashboard.py`

Additional focused validation slices now exercised during the facade conversions:

- `dev/tests/unit/apps/test_cli.py`
- `dev/tests/unit/parity/test_comparison_runtime.py`
- `dev/tests/unit/parity/test_source_selection.py`
- `dev/tests/unit/parity/test_compare_matlab_python_cli.py`
- `dev/tests/ui/test_app_layout.py`
- `dev/tests/ui/test_app_integration.py`
- `dev/tests/ui/test_share_report_entrypoint.py`

## Phase 3: Wave 2 Extraction Plan

- [ ] 8. Refactor `source/slavv/analysis/geometry.py`
  - [ ] 8.1 Extract resampling helpers
  - [ ] 8.2 Extract transform/registration helpers
  - [ ] 8.3 Preserve current geometry APIs

- [ ] 9. Refactor `source/slavv/parity/reporting.py`
  - [ ] 9.1 Extract text/markdown summary builders
  - [ ] 9.2 Extract serialization/output helpers
  - [ ] 9.3 Preserve current reporting entrypoints

- [ ] 10. Refactor `source/slavv/runtime/run_state.py`
  - [ ] 10.1 Extract snapshot read/write helpers
  - [ ] 10.2 Extract path/layout helpers
  - [ ] 10.3 Extract status rendering helpers
  - [ ] 10.4 Preserve current run-context and snapshot entrypoints

- [ ] 11. Refactor `source/slavv/parity/run_layout.py`
  - [ ] 11.1 Extract layout/path resolution helpers
  - [ ] 11.2 Extract inventory/listing helpers
  - [ ] 11.3 Extract summary/manifest helpers
  - [ ] 11.4 Preserve current run-layout-facing APIs

- [ ] 12. Validate Wave 2
  - [ ] 13.1 Run targeted subsystem tests after each module extraction
  - [ ] 13.2 Run repo lint and mypy gates
  - [ ] 13.3 Run `python -m pytest -m "unit or integration"`

## Phase 4: Wave 3 Extraction Plan

- [ ] 13. Refactor `source/slavv/core/energy.py`
  - [ ] 13.1 Extract backend-specific helpers
  - [ ] 13.2 Extract storage/resumable helpers
  - [ ] 13.3 Extract scale-selection/shared shape helpers
  - [ ] 13.4 Preserve current public energy orchestration APIs

- [ ] 14. Refactor `source/slavv/core/vertices.py`
  - [ ] 14.1 Extract MATLAB-compat painting/helpers
  - [ ] 14.2 Extract vertex-extraction helpers
  - [ ] 14.3 Preserve current public entrypoints

- [~] 15. Refactor `source/slavv/parity/metrics.py`
  - [x] 15.1 Extract vertex metric helpers
  - [x] 15.2 Extract edge metric helpers
  - [x] 15.3 Extract network/strand aggregate helpers
  - [x] 15.4 Preserve current report-facing entrypoints

- [~] 16. Refactor `source/slavv/parity/comparison.py`
  - [x] 16.1 Extract artifact discovery/loading helpers
  - [~] 16.2 Extract orchestration runtime helpers
  - [x] 16.3 Extract manifest/report assembly helpers
  - [~] 16.4 Preserve current orchestration entrypoints in `comparison.py`
  - [x] 16.5 Run focused parity comparison tests

- [ ] 17. Refactor `source/slavv/core/edge_selection.py`
  - [ ] 17.1 Extract cleanup-policy helpers
  - [ ] 17.2 Extract duplicate/conflict-resolution helpers
  - [ ] 17.3 Preserve current workflow-facing entrypoints
  - [ ] 17.4 Keep slices narrowly scoped to avoid active parity drift

- [ ] 18. Refactor `source/slavv/core/edge_candidates.py`
  - [ ] 18.1 Extract diagnostics-only helpers into a dedicated sibling module
  - [ ] 18.2 Extract terminal-resolution helpers into a dedicated sibling module
  - [ ] 18.3 Extract frontier-tracing helpers into a dedicated sibling module
  - [ ] 18.4 Keep `edge_candidates.py` as the compatibility facade
  - [ ] 18.5 Run focused core and parity regression tests

- [ ] 19. Validate Wave 3
  - [ ] 19.1 Run targeted tests for each extracted module
  - [ ] 19.2 Run repo lint gate
  - [ ] 19.3 Run repo type-check gate where applicable
  - [ ] 19.4 Run `python -m pytest -m "unit or integration"` after the parity-heavy slices

## Phase 5: Compatibility Cleanup

- [ ] 20. Audit temporary shims and re-exports
  - [ ] 20.1 List temporary compatibility imports introduced by the refactor
  - [ ] 20.2 Remove only the shims proven to be unused
  - [ ] 20.3 Keep any remaining shims documented if callers still depend on them

- [ ] 21. Align tests and docs with final ownership boundaries
  - [ ] 21.1 Move or add tests only where ownership boundaries changed materially
  - [ ] 21.2 Update contributor-facing docs if new module boundaries matter
  - [ ] 21.3 Confirm `dev/tests/README.md` placement guidance is still followed

## Phase 6: Final Verification

- [ ] 22. Run final repo verification
  - [ ] 22.1 Run `python -m compileall source dev/scripts`
  - [ ] 22.2 Run `python -m ruff format --check source dev/tests`
  - [ ] 22.3 Run `python -m ruff check source dev/tests`
  - [ ] 22.4 Run `python -m mypy`
  - [ ] 22.5 Run `python -m pytest -m "unit or integration"`

- [ ] 23. Record completion notes
  - [ ] 23.1 Record which large files were fully split
  - [ ] 23.2 Record which files were intentionally deferred and why
  - [ ] 23.3 Record any follow-up cleanup tasks that are structural but non-blocking

## Deliverables

- [x] `requirements.md` drafted
- [x] `design.md` drafted
- [x] `tasks.md` drafted
- [ ] Wave 1 completed
- [ ] Wave 2 completed
- [ ] Wave 3 completed
- [ ] Final verification completed
