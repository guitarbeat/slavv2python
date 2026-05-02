# Python Naming Guide

[Up: Reference Docs](../README.md)

Use this guide when naming or moving Python modules in the maintained SLAVV
codebase.

## Goal

Names should tell you three things quickly:

1. the pipeline domain or stage
2. the module's role in that domain
3. whether the name is preferred API, stable internal API, or compatibility-only

The rule of thumb is domain first, role second, historical detail last.

## Preferred Public Names

Use these names in maintained docs, examples, and new first-party code:

- preferred pipeline class: `source.core.SlavvPipeline`
- preferred full-run method: `run()`
- preferred energy method: `compute_energy()`
- preferred network builder method: `build_network()`
- preferred final-stage term: `network`, not `graph`

Example:

```python
from source.core import SlavvPipeline

pipeline = SlavvPipeline()
results = pipeline.run(image, parameters)
```

## Stable Internal Names

Use domain-first package names for maintained internal surfaces:

- `source.core.network`
- `source.core.energy_internal`
- `source.core.vertices_internal`
- `source.core.edges_internal`
- `source.runtime.run_tracking`
- `source.workflows.pipeline_session`
- `source.workflows.pipeline_execution`
- `source.workflows.stage_resolution`
- `source.workflows.stage_artifacts`
- `source.apps.cli`
- `source.apps.streamlit`

Within a stage package, prefer role names such as:

- `candidate_generation`
- `edge_tracing`
- `edge_selection`
- `bridge_insertion`
- `vertex_selection`
- `vertex_painting`
- `energy_chunking`
- `energy_provenance`

Avoid vague names like `standard`, `common`, or `payloads` unless the module is
truly cross-cutting and the name is still precise.

## Compatibility Names

These names still work for one migration cycle, but they are not the preferred
surface for new code:

- `source.core.SLAVVProcessor`
- `process_image()`
- `calculate_energy_field()`
- `construct_network()`
- `source.core.graph`
- `source.runtime._run_state.*`
- flat app modules such as `source.apps.cli_parser` or `source.apps.web_app_*`

Compatibility names may emit `DeprecationWarning` in Python code. Public CLI
behavior should stay quiet.

## Naming Rules

- Prefer domain nouns first: `energy`, `vertices`, `edges`, `network`,
  `analysis`, `visualization`, `run_tracking`.
- Prefer role names second: `pipeline`, `runner`, `state`, `services`, `io`,
  `selection`, `finalize`, `resumable`.
- Do not use leading underscores for package names that are part of normal
  day-to-day development.
- Keep parity-specific names explicit under `matlab_compat`.
- When in doubt, optimize for stack traces and grep results that read like the
  pipeline itself.

## Migration Policy

- Keep the `source` package root stable for now.
- Keep `slavv` and `slavv-app` entrypoints stable.
- New maintained docs and examples should teach the preferred names first.
- Old import paths can remain as shims during the compatibility window, but new
  first-party call sites should not move back to them.
