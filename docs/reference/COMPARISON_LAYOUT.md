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

## Recommended Output Root

For live MATLAB-enabled comparisons, prefer a local non-synced drive with
comfortable free space. Good examples are:

- Windows: `D:\slavv_comparisons\20260401_parity`
- POSIX: `/tmp/slavv_comparisons/20260401_parity`

Avoid placing fresh MATLAB outputs under OneDrive-synced folders, network
mounts, or repo-local scratch paths unless you are intentionally debugging a
failure mode. The comparison preflight now records that decision in
`99_Metadata/output_preflight.json`, but the safest default is still an
explicit local output root outside the repository.

## Typical Contents

### `01_Input/`

- source image path or copied input manifest
- normalized comparison parameters
- `matlab_results/batch_*` directories from MATLAB execution
- imported MATLAB artifacts that act as upstream inputs to Python parity reruns

### `02_Output/`

- `python_results/checkpoints/`
- `python_results/network.json`, `python_results/vertices.json`
- `python_results/stages/edges/candidates.pkl`
- `python_results/stages/edges/candidate_audit.json` (candidate source breakdown and origin-level counters)
- resumable pipeline state and stage snapshots

### `03_Analysis/`

- `summary.txt`
- `comparison_report.json`
- report assets produced by evaluation and reporting helpers
- candidate-audit provenance path references shown in `summary.txt` for quick triage

### `99_Metadata/`

- `comparison_params.normalized.json`
- `run_snapshot.json`
- `output_preflight.json`
- `matlab_status.json`
- `matlab_failure_summary.json` when MATLAB fails
- `run_manifest.md`

## Metadata Contract

The staged comparison workflow now treats `99_Metadata/` as the shared
human-facing ledger for orchestration decisions:

| Artifact | Purpose |
| --- | --- |
| `comparison_params.normalized.json` | Normalized parameter payload passed to both MATLAB and Python so reruns can inspect the exact effective settings |
| `run_snapshot.json` | Shared run-state ledger with overall status, current stage, and optional-task artifacts such as `output_preflight` and `matlab_status` |
| `output_preflight.json` | Authoritative preflight decision for the selected output root, including warnings, fatal errors, free-space estimates, and recommended action |
| `matlab_status.json` | Normalized MATLAB rerun semantics such as selected `batch_*` folder, resume mode, next stage, partial-artifact detection, and predicted rerun behavior |
| `matlab_failure_summary.json` | Concise failure summary and log-tail evidence persisted when MATLAB exits unsuccessfully |
| `run_manifest.md` | Human-readable run summary that links the preflight decision, resume semantics, authoritative files, and comparison outputs |

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

As of April 1, 2026, the staged comparison surface is also the expected place to
inspect output-root preflight decisions and MATLAB rerun semantics. Exact
vertex parity is in place, and the parity-only edge path now records richer
candidate provenance and frontier diagnostics. Final exact edge/strand
confirmation still comes from a fresh live MATLAB-enabled comparison run on a
healthy local output root.
