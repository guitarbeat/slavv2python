# Python Naming Guide

[Up: Reference Docs](../README.md)

This guide defines naming conventions for Python modules in `slavv_python`.

## Goal
Module names should indicate:
1. Pipeline domain or stage (first)
2. Module role (second)
3. API classification (public, internal, or compatibility) (last)

## Preferred Public Names

Use these names in maintained docs, examples, and new first-party code:

- preferred pipeline class: `slavv_python.engine.SlavvPipeline` (also re-exported as `slavv_python.SlavvPipeline`)
- preferred full-run method: `run()`
- preferred energy method: `compute_energy()`
- preferred network builder method: `build_network()`
- preferred final-stage term: `network`, not `graph`

Example:

```python
from slavv_python import SlavvPipeline

pipeline = SlavvPipeline()
results = pipeline.run(image, parameters)  # PipelineResult; mapping-compatible

# Prefer typed access in new code:
vertices = results.vertices
edges = results.edges
```

### UI session envelope (`AppRunState`)

Streamlit and shared-state code should hold a typed run envelope, not a plain dict:

- preferred schema: `slavv_python.schema.app_run.AppRunState`
- preferred session accessor: `slavv_python.interface.shared_state.get_app_run(session)`
- preferred storage: `session_state["processing_results"]` as `AppRunState` (wraps `PipelineResult`)
- defer dict serialization to export/share boundaries only (`normalize_state_results`, `exports`, `share_report`)

Example:

```python
from slavv_python.interface.shared_state import get_app_run, store_processing_session_state

store_processing_session_state(session_state, pipeline_result, parameters=parameters)
app_run = get_app_run(st.session_state)
vertices = app_run.pipeline.vertices  # typed; Mapping access also works on AppRunState
```

## Stable Internal Names

Use domain-first package names for maintained internal surfaces:

- `slavv_python.pipeline.network`
- `slavv_python.pipeline.energy`
- `slavv_python.pipeline.vertices`
- `slavv_python.pipeline.edges`
- `slavv_python.engine.state` (`run_ledger`, `stage_handle`)
- `slavv_python.schema.app_run` (`AppRunState`)
- `slavv_python.workflows.pipeline`
- `slavv_python.workflows.pipeline_setup`
- `slavv_python.workflows.pipeline_stages`
- `slavv_python.workflows.profiles`
- `slavv_python.interface.cli`
- `slavv_python.interface.streamlit`

Within a stage package, prefer role names such as:

- `discovery` (strategy seam for edge candidate generation)
- `manager` (stage facade: `VertexManager`, `EdgeManager`, `NetworkManager`; `run()` + `run_resumable()`)
- `detection` (vertex candidate scan/crop/choose)
- `candidate_generation`
- `tracing`
- `selection`
- `bridge_insertion`
- `painting`
- `chunking`
- `provenance`

Avoid vague names like `standard`, `common`, or `payloads` unless the module is
truly cross-cutting and the name is still precise.

## Compatibility Policy

The codebase has been modernized to remove legacy shims. Previous names such as
`SLAVVProcessor`, `process_image()`, and `slavv_python.core.graph` have been
fully retired.

### Stable Entrypoints
- `slavv_python.engine.SlavvPipeline` (re-exported from `slavv_python`)
- `run()`
- `compute_energy()`
- `build_network()`
- `slavv_python.pipeline.network`

## Naming Rules

- Prefer domain nouns first: `energy`, `vertices`, `edges`, `network`,
  `analysis`, `visualization`, `run_tracking`.
- Prefer role names second: `pipeline`, `runner`, `state`, `services`, `io`,
  `selection`, `finalize`, `resumable`.
- Do not use leading underscores for package names that are part of normal
  day-to-day development.
- When in doubt, optimize for stack traces and grep results that read like the
  pipeline itself.

## MATLAB Parity Filename Convention

Use filenames that state **role** and **MATLAB lineage** so a developer can grep
one `.m` basename and land on the matching Python port.

| Prefix | Meaning | Example |
| --- | --- | --- |
| `matlab_<function>_<version>.py` | Direct port of a named MATLAB function | `matlab_get_energy_v202_chunked.py` ← `get_energy_V202.m` |
| `matlab_<function>_<version>_<facet>.py` | Facet of the same MATLAB function | `matlab_energy_filter_v200.py` ← `energy_filter_V200.m` |
| `matlab_<concept>_<facet>.py` | Shared helper in a MATLAB algorithm family | `matlab_watershed_heap.py` (watershed frontier heap) |
| `matlab_<function>.py` | Port of a named MATLAB helper (no version suffix) | `matlab_calculate_linear_strel_range.py` ← `calculate_linear_strel_range.m` |
| `matlab_<function>_<facet>.py` | Geometry/facet of a versioned MATLAB function | `matlab_get_edges_v300_geometry.py` ← `get_edges_V300.m` |
| `parity_<domain>_<tool>.py` | Parity harness / probe (not a MATLAB port) | `parity_energy_voxel_probe.py` |
| `parity_probe_<experiment>.py` | Batch parity experiment driver under `tests/support/` | `parity_probe_scale_winner.py` |

**Rules:**

- Put the MATLAB function basename in the Python filename when the module is a
  port (`get_energy_V202` → `matlab_get_energy_v202_*`).
- Suffix the Python role when one `.m` file spans multiple modules (`_chunked`,
  `_filter`, `_heap`, `_frontier`).
- Start module docstrings with `MATLAB port:` or `Parity diagnostic:` plus the
  canonical path under `external/Vectorization-Public/`.
- Keep stage facades role-named (`manager.py`, `discovery.py`, `selection.py`);
  do not embed MATLAB version numbers in orchestration modules.
- See [MATLAB_PARITY_MAPPING.md](../core/MATLAB_PARITY_MAPPING.md) for the full
  function-to-file table.

## Migration Policy

- Keep the `slavv_python` package root stable for now.
- Keep `slavv` and `slavv-app` entrypoints stable.
- New maintained docs and examples should teach the preferred names first.
- Old import paths can remain as shims during the compatibility window, but new
  first-party call sites should not move back to them.
