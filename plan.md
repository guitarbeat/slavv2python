# Large-Module Refactor Plan

## Purpose

This file tracks the current state of the large-module breakdown work in
`slavv2python`. It is meant to answer four practical questions:

1. What has already been refactored successfully?
2. What is still large?
3. What validation has been keeping the refactor safe?
4. What should happen next?

## Strategy

The working pattern has been:

- keep the original module path stable
- turn the original large file into a thin facade when tests or callers rely on
  that path
- move real implementation into sibling modules or a private package
- preserve monkeypatch-sensitive names on the facade when tests depend on them
- validate each slice with focused tests, then `ruff`, then `mypy`

This approach has worked better than purely cosmetic cleanup because it
produces real line-count reduction in the original files without changing the
public import surface.

## Progress So Far

### Fully-landed facade/package conversions

- `source/slavv/apps/web_app.py`
  - now a thin facade
  - page logic moved into:
    - `source/slavv/apps/web_app_dashboard_page.py`
    - `source/slavv/apps/web_app_processing_page.py`
    - `source/slavv/apps/web_app_curation_page.py`
    - `source/slavv/apps/web_app_visualization_page.py`
    - `source/slavv/apps/web_app_analysis_page.py`
    - `source/slavv/apps/web_app_static_pages.py`
    - `source/slavv/apps/web_app_shell.py`
- `source/slavv/visualization/network_plots.py`
  - now a thin facade
  - plotting logic moved into:
    - `source/slavv/visualization/network_plot_spatial_2d.py`
    - `source/slavv/visualization/network_plot_spatial_3d.py`
    - `source/slavv/visualization/network_plot_statistics.py`
  - shared helpers live in:
    - `source/slavv/visualization/network_plot_helpers.py`
    - `source/slavv/visualization/network_plot_layout.py`
    - `source/slavv/visualization/network_plot_dashboard.py`
    - `source/slavv/visualization/network_plot_exports.py`
- `source/slavv/core/edge_candidates.py`
  - now a thin facade over `source/slavv/core/_edge_candidates/`
- `source/slavv/apps/cli.py`
  - now a thin facade over:
    - `source/slavv/apps/cli_parser.py`
    - `source/slavv/apps/cli_shared.py`
    - `source/slavv/apps/cli_exported_network.py`
    - `source/slavv/apps/cli_commands.py`
- `source/slavv/analysis/ml_curator.py`
  - reduced by moving major helper and curator responsibilities into sibling
    modules

### Partial facade/package conversions in progress

- `source/slavv/parity/comparison.py`
  - now delegates into `source/slavv/parity/_comparison/`
  - extracted modules currently include:
    - `analysis.py`
    - `artifacts.py`
    - `config.py`
    - `health_check.py`
    - `matlab_runner.py`
    - `python_runner.py`
    - `python_sources.py`
    - `reuse.py`
    - `standalone.py`
    - `task_recording.py`
- `source/slavv/parity/metrics.py`
  - initial split started under `source/slavv/parity/_metrics/`

## Current Large Files

These are the current `source/` Python files above `500` lines:

- `1054` `source/slavv/parity/metrics.py`
- `1030` `source/slavv/core/energy.py`
- `978` `source/slavv/parity/comparison.py`
- `904` `source/slavv/runtime/run_state.py`
- `836` `source/slavv/parity/run_layout.py`
- `697` `source/slavv/core/vertices.py`
- `677` `source/slavv/analysis/geometry.py`
- `653` `source/slavv/core/edge_selection.py`
- `630` `source/slavv/parity/workflow_assessment.py`
- `613` `source/slavv/parity/reporting.py`
- `579` `source/slavv/analysis/ml_curator.py`
- `559` `source/slavv/core/edge_primitives.py`
- `554` `source/slavv/core/edges.py`

## Validation Pattern

The refactor has been safest when each slice follows this order:

1. Move one coherent responsibility block.
2. Preserve facade-level names that tests monkeypatch or import directly.
3. Run the nearest focused tests first.
4. Run `python -m ruff check ...`.
5. Run `python -m mypy`.

Examples of focused suites that have already been useful:

- app/UI:
  - `dev/tests/unit/apps/test_web_app_dashboard.py`
  - `dev/tests/unit/apps/test_web_app_dashboard_refactor.py`
  - `dev/tests/unit/apps/test_web_app_artifacts_refactor.py`
  - `dev/tests/ui/test_app_layout.py`
  - `dev/tests/ui/test_app_integration.py`
  - `dev/tests/ui/test_share_report_entrypoint.py`
- parity/comparison:
  - `dev/tests/unit/parity/test_comparison_runtime.py`
  - `dev/tests/unit/parity/test_source_selection.py`
  - `dev/tests/unit/parity/test_compare_matlab_python_cli.py`
- visualization:
  - `dev/tests/ui/test_visualization_aspect.py`
  - `dev/tests/ui/test_network_slice.py`
  - `dev/tests/ui/test_edge_coloring.py`
  - `dev/tests/ui/test_flow_field.py`
  - `dev/tests/ui/test_summary_dashboard.py`
  - `dev/tests/ui/test_visualization_exports.py`

## What Has Worked Well

- Moving whole responsibility blocks works better than tiny helper nibbles.
- Thin facades are especially effective when tests monkeypatch module-level
  names.
- Private packages are a good fit for parity-heavy or orchestration-heavy
  modules.
- Directly measuring the original file size after each slice keeps the work
  honest.

## Current In-Progress State

At the time this file was written, the active in-progress refactor was:

- `source/slavv/parity/comparison.py`
- `source/slavv/parity/_comparison/config.py`

That means `comparison.py` is already below `1000` lines, but it is still the
active parity-heavy facade being reduced further.

## Recommended Next Order

1. Finish `source/slavv/parity/metrics.py`
   - it is now the largest remaining file
   - an `_metrics` package already exists, so the seam has been proven
2. Then refactor `source/slavv/core/energy.py`
   - large, but structurally cleaner than the parity files
3. Then tackle `source/slavv/runtime/run_state.py`
   - important compatibility surface, but good candidate for facade + package
4. After that, continue with:
   - `source/slavv/parity/run_layout.py`
   - `source/slavv/parity/reporting.py`
   - `source/slavv/parity/workflow_assessment.py`
5. Finish the remaining core payload stack:
   - `vertices.py`
   - `edge_selection.py`
   - `edge_primitives.py`
   - `edges.py`

## Near-Term Goal

Bring the remaining `source/` files over `500` lines down one by one using the
same facade/package pattern, without regressing the current parity and app test
surfaces.
