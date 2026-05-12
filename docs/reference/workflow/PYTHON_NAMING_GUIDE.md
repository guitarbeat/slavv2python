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

- preferred pipeline class: `slavv_python.core.SlavvPipeline`
- preferred full-run method: `run()`
- preferred energy method: `compute_energy()`
- preferred network builder method: `build_network()`
- preferred final-stage term: `network`, not `graph`

Example:

```python
from slavv_python.core import SlavvPipeline

pipeline = SlavvPipeline()
results = pipeline.run(image, parameters)
```

## Stable Internal Names

Use domain-first package names for maintained internal surfaces:

- `slavv_python.core.network`
- `slavv_python.core.energy`
- `slavv_python.core.vertices`
- `slavv_python.core.edges`
- `slavv_python.runtime.run_tracking`
- `slavv_python.workflows.pipeline_session`
- `slavv_python.workflows.pipeline_execution`
- `slavv_python.workflows.stage_resolution`
- `slavv_python.workflows.stage_artifacts`
- `slavv_python.apps.cli`
- `slavv_python.apps.streamlit`

Within a stage package, prefer role names such as:

- `candidate_generation`
- `tracing`
- `selection`
- `bridge_insertion`
- `painting`
- `chunking`
- `provenance`

Avoid vague names like `standard`, `common`, or `payloads` unless the module is
truly cross-cutting and the name is still precise.

## Compatibility Names

These names still work for one migration cycle, but they are not the preferred
surface for new code:

- `slavv_python.core.SLAVVProcessor`
- `process_image()`
- `calculate_energy_field()`
- `construct_network()`
- `slavv_python.core.graph`
- `slavv_python.runtime._run_state.*`
- flat app modules such as `slavv_python.apps.cli_parser` or `slavv_python.apps.web_app_*`

Compatibility names may emit `DeprecationWarning` in Python code. Public CLI
behavior should stay quiet.

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
