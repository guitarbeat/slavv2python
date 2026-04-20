# WS0 Safety Net Checklist

## Purpose

This file is the actionable starter packet for `WS0 - Safety Net And Shared
Builders` from `MODULARIZATION_PARALLEL_PLAN.md`.

The goal of `WS0` is to make the later modularization work safer and easier to
split across multiple parallel PRs. This workstream should avoid production code
changes whenever possible. It is primarily about:

- improving test ownership
- introducing reusable test builders
- renaming task-history tests to behavior names
- moving tests away from convenience locations and toward owning package
  surfaces

## Non-Goals

`WS0` should not:

- change public runtime behavior
- narrow public imports
- move major production modules
- consolidate format ownership
- rewrite the Streamlit app or CLI

Those belong to later workstreams.

## Current Pain Points

The current test tree already exposes the modularity friction:

- `dev/tests/unit/io/test_safe_unpickle.py` exercises
  `slavv.utils.safe_unpickle`, so the test owner does not match the production
  owner.
- `dev/tests/unit/io/test_casx_export.py`,
  `dev/tests/unit/io/test_vmv_export.py`, and part of
  `dev/tests/unit/io/test_mat_io.py` exercise
  `slavv.visualization.NetworkVisualizer`, so the current location suggests `io`
  ownership while the tests actually cover a visualization wrapper.
- `dev/tests/unit/apps/test_web_app_dashboard_refactor.py` and
  `dev/tests/unit/apps/test_web_app_artifacts_refactor.py` use task-history
  naming rather than behavior naming.
- `dev/tests/unit/runtime/test_run_state.py` mixes pure runtime tests with
  tests that go through `SLAVVProcessor`, which makes failures harder to triage.

## Deliverables

This workstream is done when the following exist:

- `dev/tests/support/`
- reusable fixture builders for payloads, networks, and run-state snapshots
- test filenames that reflect behavior and ownership
- a smaller, better-partitioned runtime test surface
- a documented set of focused test commands for later workstreams

## Proposed New Test Support Package

Create:

- `dev/tests/support/__init__.py`
- `dev/tests/support/payload_builders.py`
- `dev/tests/support/network_builders.py`
- `dev/tests/support/run_state_builders.py`

Suggested responsibilities:

### `payload_builders.py`

Provide helpers like:

- `build_energy_result(...)`
- `build_vertices_payload(...)`
- `build_edges_payload(...)`
- `build_network_payload(...)`
- `build_processing_results(...)`

These should build stable synthetic data structures that later workstreams can
 reuse while the repo still uses dict-based payloads.

### `network_builders.py`

Provide helpers like:

- `build_network_object(...)`
- `build_export_ready_processing_results(...)`
- `build_minimal_network_json_payload(...)`

These should reduce duplication across export, visualization, and I/O tests.

### `run_state_builders.py`

Provide helpers like:

- `build_run_context(...)`
- `build_snapshot_dict(...)`
- `build_stage_snapshot_dict(...)`
- `build_optional_task_dict(...)`
- `materialize_legacy_checkpoint_surface(...)`

These should let runtime and parity tests prepare snapshots without relying on
`SLAVVProcessor` for setup.

## Exact Planned File Moves And Renames

These are the recommended `WS0` moves. They are grouped by confidence level so
parallel contributors can work without blocking each other.

### Pure ownership move

- move
  `dev/tests/unit/io/test_safe_unpickle.py`
  to
  `dev/tests/unit/utils/test_safe_unpickle.py`

Reason:

- the production owner is `slavv.utils.safe_unpickle`

### Behavior-name renames

- rename
  `dev/tests/unit/apps/test_web_app_dashboard_refactor.py`
  to
  `dev/tests/unit/apps/test_web_app_dashboard_facade.py`
- rename
  `dev/tests/unit/apps/test_web_app_artifacts_refactor.py`
  to
  `dev/tests/unit/apps/test_web_app_artifacts_facade.py`

Reason:

- these tests validate facade/import contracts, not a historical refactor task

### Split mixed-owner MAT/visualizer tests

- split
  `dev/tests/unit/io/test_mat_io.py`
  into:
  - `dev/tests/unit/io/test_mat_network_io.py`
  - `dev/tests/unit/visualization/test_network_visualizer_mat_export.py`

Suggested split:

- keep these in `test_mat_network_io.py`
  - `test_mat_roundtrip`
  - `test_load_empty_mat_network_shapes`
- move these to `test_network_visualizer_mat_export.py`
  - `test_mat_export_via_visualizer`
  - `test_mat_export_complex_params`
  - `test_json_export_handles_numpy_scalars_and_paths`

Reason:

- the moved tests primarily validate `NetworkVisualizer().export_network_data`

### Move visualization-wrapper export tests out of `io`

- move
  `dev/tests/unit/io/test_casx_export.py`
  to
  `dev/tests/unit/visualization/test_network_visualizer_casx_export.py`
- move
  `dev/tests/unit/io/test_vmv_export.py`
  to
  `dev/tests/unit/visualization/test_network_visualizer_vmv_export.py`

Reason:

- they currently exercise `slavv.visualization.NetworkVisualizer`
- true format ownership cleanup belongs later in `WS7`
- for now the test location should match the code under test

### Split the large runtime test file

Split:

- `dev/tests/unit/runtime/test_run_state.py`

Into:

- `dev/tests/unit/runtime/test_run_state_atomic_io.py`
- `dev/tests/unit/runtime/test_run_context_lifecycle.py`
- `dev/tests/unit/runtime/test_resume_guard.py`
- `dev/tests/unit/runtime/test_legacy_snapshot_bootstrap.py`
- `dev/tests/unit/runtime/test_status_rendering.py`
- `dev/tests/integration/test_pipeline_run_state_integration.py`

Suggested test moves:

- `test_atomic_write_json_replaces_previous_content`
  -> `test_run_state_atomic_io.py`
- `test_run_context_persists_snapshot_lifecycle`
  -> `test_run_context_lifecycle.py`
- `test_resume_guard_blocks_mismatched_input`
  -> `test_resume_guard.py`
- `test_force_rerun_from_energy_resets_pipeline_state`
  -> `test_resume_guard.py`
- `test_legacy_checkpoints_bootstrap_snapshot`
  -> `test_legacy_snapshot_bootstrap.py`
- `test_from_existing_preserves_existing_target_stage`
  -> `test_run_context_lifecycle.py`
- `test_load_legacy_run_snapshot_is_read_only`
  -> `test_legacy_snapshot_bootstrap.py`
- `test_parse_time_uses_utc_epoch`
  -> `test_status_rendering.py` or a small helper-focused runtime file
- `test_build_status_lines_includes_matlab_resume_details`
  -> `test_status_rendering.py`
- `test_process_image_blocks_reuse_when_parameters_change`
  -> `test_pipeline_run_state_integration.py`
- `test_process_image_rejects_invalid_stop_after`
  -> `test_pipeline_run_state_integration.py`
- `test_preprocess_failure_marks_run_failed`
  -> `test_pipeline_run_state_integration.py`

Reason:

- processor-mediated tests are integration coverage, not pure runtime unit
  coverage

## Starter PR Boundaries

These PR boundaries are designed to minimize conflicts.

### PR A - Add shared builders only

Write scope:

- `dev/tests/support/`
- minimal import updates in a tiny number of tests

Deliverables:

- new support package
- no test moves yet

Recommended validation:

```powershell
python -m pytest dev/tests/unit/parity/test_workflow_assessment.py
python -m pytest dev/tests/unit/runtime/test_run_state.py
python -m pytest dev/tests/unit/apps/test_web_app_dashboard.py
```

### PR B - Ownership moves and behavior-name renames

Write scope:

- `dev/tests/unit/utils/`
- `dev/tests/unit/apps/`
- file renames only

Deliverables:

- `test_safe_unpickle.py` moved to `unit/utils`
- app facade tests renamed

Recommended validation:

```powershell
python -m pytest dev/tests/unit/utils/test_safe_unpickle.py
python -m pytest dev/tests/unit/apps/test_web_app_dashboard_facade.py
python -m pytest dev/tests/unit/apps/test_web_app_artifacts_facade.py
```

### PR C - Visualization-wrapper export test relocation

Write scope:

- `dev/tests/unit/io/`
- `dev/tests/unit/visualization/`
- optional shared builders from `dev/tests/support/`

Deliverables:

- CASX and VMV wrapper tests moved to visualization
- `test_mat_io.py` split by owner

Recommended validation:

```powershell
python -m pytest dev/tests/unit/io/test_mat_network_io.py
python -m pytest dev/tests/unit/visualization/test_network_visualizer_mat_export.py
python -m pytest dev/tests/unit/visualization/test_network_visualizer_casx_export.py
python -m pytest dev/tests/unit/visualization/test_network_visualizer_vmv_export.py
```

### PR D - Runtime test partitioning

Write scope:

- `dev/tests/unit/runtime/`
- `dev/tests/integration/`
- optional builders in `dev/tests/support/`

Deliverables:

- runtime tests split by responsibility
- processor-mediated tests moved to integration

Recommended validation:

```powershell
python -m pytest dev/tests/unit/runtime/
python -m pytest dev/tests/integration/test_pipeline_run_state_integration.py
```

## Parallel Ownership Map

To avoid conflicts, contributors should claim one of these write zones:

- Zone 1
  - `dev/tests/support/`
- Zone 2
  - `dev/tests/unit/utils/`
  - `dev/tests/unit/apps/`
- Zone 3
  - `dev/tests/unit/io/`
  - `dev/tests/unit/visualization/`
- Zone 4
  - `dev/tests/unit/runtime/`
  - `dev/tests/integration/`

Only one branch should move files in a zone at a time unless the changes are
explicitly coordinated.

## Suggested Builder Adoption Targets

The following existing tests should be the first to adopt shared builders once
they exist:

- `dev/tests/unit/apps/test_curation_state.py`
- `dev/tests/unit/apps/test_share_report.py`
- `dev/tests/unit/io/test_network_io.py`
- `dev/tests/unit/io/test_exporters.py`
- `dev/tests/unit/runtime/test_run_state.py` or its split descendants
- `dev/tests/unit/parity/test_workflow_assessment.py`
- `dev/tests/unit/parity/test_matlab_status.py`

These tests currently construct payloads or snapshots inline and are likely to
benefit immediately from shared helpers.

## Focused Commands For Later Workstreams

Once `WS0` lands, these commands should become the default smoke tests for
later modularization PRs:

### Workflow and runtime seams

```powershell
python -m pytest dev/tests/unit/runtime/
python -m pytest dev/tests/integration/test_pipeline_run_state_integration.py
```

### App and facade seams

```powershell
python -m pytest dev/tests/unit/apps/test_cli.py
python -m pytest dev/tests/unit/apps/test_web_app_dashboard.py
python -m pytest dev/tests/unit/apps/test_web_app_dashboard_facade.py
python -m pytest dev/tests/unit/apps/test_web_app_artifacts_facade.py
```

### IO and visualization seams

```powershell
python -m pytest dev/tests/unit/io/
python -m pytest dev/tests/unit/visualization/
```

### Parity-sensitive seams

```powershell
python -m pytest dev/tests/unit/parity/
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

## Done Criteria

`WS0` is done when:

- `dev/tests/support/` exists and is being used by at least a few tests
- the obvious ownership leaks listed above are corrected
- the task-history test names are gone
- runtime unit tests and processor integration tests are separated
- contributors can pick later workstreams by zone without first untangling the
  test tree

## Recommended Execution Order

1. PR A - shared builders
2. PR B - simple ownership moves and renames
3. PR C - visualization-wrapper export test relocation
4. PR D - runtime test partitioning

This order keeps the early changes low risk and gives the later PRs common
helpers to build on.
