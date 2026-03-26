# Comparison Run Layout

This document describes the canonical staged layout for generated MATLAB/Python
comparison runs. It replaces the older `workspace/experiments` note, which no
longer matches the current repository structure.

## Goals

- Keep generated comparison artifacts out of the package source tree.
- Make resumed runs and manual inspections predictable.
- Separate upstream inputs, Python outputs, analysis artifacts, and run metadata.
- Preserve a layout that works for both full comparisons and parity-focused
  reruns.

## Canonical Layout

Use a timestamped run root under the user-selected output directory. Inside that
run root, keep the following staged folders:

| Path | Purpose |
| --- | --- |
| `01_Input/` | Immutable inputs for the run, including copied parameters and MATLAB batch artifacts |
| `02_Output/` | Python processing outputs such as checkpoints, exported networks, and intermediate artifacts |
| `03_Analysis/` | Comparison summaries, `comparison_report.json`, rendered tables, and other human-facing analysis |
| `99_Metadata/` | Run manifests, resume state, status snapshots, and other orchestration metadata |

## Typical Contents

### `01_Input/`

- source image path or copied input manifest
- normalized comparison parameters
- `matlab_results/batch_*` directories from MATLAB execution
- imported MATLAB artifacts that act as upstream inputs to Python parity reruns

### `02_Output/`

- `python_results/checkpoints/`
- exported `json` or `csv` network outputs
- resumable pipeline state and stage snapshots

### `03_Analysis/`

- `summary.txt`
- `comparison_report.json`
- report assets produced by evaluation and reporting helpers

### `99_Metadata/`

- run manifest
- status snapshots
- resume guards and other orchestration metadata

## Workflow Notes

- `workspace/scripts/cli/compare_matlab_python.py` should treat this staged
  layout as the default organization for generated runs.
- `slavv import-matlab` imports a MATLAB batch into checkpoint-compatible
  artifacts that can then be consumed by parity reruns from the `edges` or
  `network` stage.
- Low-level standalone comparison helpers may still emit analysis artifacts at
  the caller-selected root when invoked directly. Prefer the staged layout when
  building durable comparison outputs for inspection.

## Repository Guidance

- Treat generated outputs under `comparisons/`, `comparison_output*/`, ad-hoc
  temp folders, and cache directories as disposable artifacts rather than
  source inputs for code changes.
- Keep documentation and code references pointed at the staged layout, not at
  one-off local output paths.
- If a workflow produces a new durable output convention, update this document
  and the relevant tests before relying on it.

## Current Parity Context

As of March 26, 2026, the staged comparison surface is being used to validate
MATLAB-energy parity work. Exact vertex parity is in place, and the parity-only
edge-cleanup plus strand-construction path has been tightened to follow MATLAB
ordering more closely. Final exact edge/strand confirmation still comes from a
live MATLAB-enabled comparison run.
