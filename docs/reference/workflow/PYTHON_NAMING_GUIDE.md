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

## Stable Internal Names

Use domain-first package names for maintained internal surfaces:

- `slavv_python.processing.stages.network`
- `slavv_python.processing.stages.energy`
- `slavv_python.processing.stages.vertices`
- `slavv_python.processing.stages.edges`
- `slavv_python.engine.state`
- `slavv_python.workflows.pipeline`
- `slavv_python.workflows.pipeline_setup`
- `slavv_python.workflows.pipeline_stages`
- `slavv_python.workflows.profiles`
- `slavv_python.interface.cli`
- `slavv_python.interface.streamlit`

Within a stage package, prefer role names such as:

- `discovery` (strategy seam for edge candidate generation)
- `manager` (stage facade for resumable edge lifecycle)
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
- `slavv_python.processing.stages.network`

## Naming Rules

- Prefer domain nouns first: `energy`, `vertices`, `edges`, `network`,
  `analysis`, `visualization`, `run_tracking`.
- Prefer role names second: `pipeline`, `runner`, `state`, `services`, `io`,
  `selection`, `finalize`, `resumable`.
- Do not use leading underscores for package names that are part of normal
  day-to-day development.
- Keep parity-specific names explicit under `matlab_algorithms`.
- When in doubt, optimize for stack traces and grep results that read like the
  pipeline itself.

## Migration Policy

- Keep the `slavv_python` package root stable for now.
- Keep `slavv` and `slavv-app` entrypoints stable.
- New maintained docs and examples should teach the preferred names first.
- Old import paths can remain as shims during the compatibility window, but new
  first-party call sites should not move back to them.
