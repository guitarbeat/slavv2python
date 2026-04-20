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

## Completed

The main `WS0` safety-net work is now in place:

- `dev/tests/support/` exists with payload, network, and run-state builders
- ownership leaks were corrected for utils and visualization-wrapper tests
- task-history app test names were replaced with behavior names
- MAT/JSON visualizer export coverage was consolidated under the visualization
  owner surface
- the large runtime test file was split into focused runtime unit files plus
  `dev/tests/integration/test_pipeline_run_state_integration.py`
- shared builders are now used in several app, io, runtime, visualization, and
  parity tests

## Current Baseline

The planning items above are now complete. Treat the following as the default
WS0 safety-net baseline for later work:

- `dev/tests/support/` for reusable builders
- `dev/tests/unit/runtime/` for runtime-owned unit seams
- `dev/tests/integration/test_pipeline_run_state_integration.py` for
  processor-mediated run-state coverage
- `dev/tests/unit/visualization/` for `NetworkVisualizer` wrapper exports
- `dev/tests/unit/utils/` for `safe_unpickle`
- `dev/tests/support/payload_builders.py` for synthetic processing payloads
- `dev/tests/support/network_builders.py` for reusable network objects and
  export fixtures
- `dev/tests/support/run_state_builders.py` for snapshot and checkpoint setup

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

## Status

`WS0` is complete enough to use as the baseline for `WS1+` workstreams.
