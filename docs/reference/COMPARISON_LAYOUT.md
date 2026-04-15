# Comparison Run Layout

This document describes the canonical staged layout for generated MATLAB/Python
comparison runs. It replaces the older `dev/experiments` note, which no
longer matches the current repository structure.

## Goals

- Keep generated comparison artifacts out of the package source tree.
- Make resumed runs and manual inspections predictable.
- Separate upstream inputs, Python outputs, analysis artifacts, and run metadata.
- Preserve a layout that works for both full comparisons and parity-focused
  reruns.
- Keep this file focused on the durable folder contract; chapter-specific
  workflow notes belong in the relevant chapter docs.

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
- `shared_neighborhood_diagnostics.json` - neighborhood-level diagnostic report for edge parity gaps
- `shared_neighborhood_diagnostics.md` - human-readable diagnostic report
- `proof_artifacts/` - directory containing stage-isolated network gate proof artifacts
  - `network_gate_proof_{timestamp}.json` - proof artifact in JSON format
  - `network_gate_proof_{timestamp}.md` - proof artifact in markdown format
- `proof_artifact_index.json` - index of all proof artifacts with timestamps and parity status

### `99_Metadata/`

- `comparison_params.normalized.json`
- `run_snapshot.json`
- `output_preflight.json`
- `matlab_status.json`
- `matlab_failure_summary.json` when MATLAB fails
- `run_manifest.md`
- `loop_assessment.json` - workflow assessment report with reuse eligibility
- `network_gate_validation.json` - pre-execution validation for stage-isolated network gate
- `network_gate_execution.json` - execution metadata and timing for network gate runs

## Metadata Contract

The staged comparison workflow now treats `99_Metadata/` as the shared
human-facing ledger for orchestration decisions:

| Artifact | Purpose |
| --- | --- |
| `comparison_params.normalized.json` | Normalized parameter payload passed to both MATLAB and Python so reruns can inspect the exact effective settings |
| `run_snapshot.json` | Shared run-state ledger with overall status, current stage, and optional-task artifacts such as `output_preflight`, `matlab_status`, and `matlab_pipeline` launch/reuse decisions |
| `output_preflight.json` | Authoritative preflight decision for the selected output root, including warnings, fatal errors, free-space estimates, and recommended action |
| `matlab_status.json` | Normalized MATLAB rerun semantics such as selected `batch_*` folder, resume mode, next stage, partial-artifact detection, and predicted rerun behavior |
| `matlab_failure_summary.json` | Concise failure summary and log-tail evidence persisted when MATLAB exits unsuccessfully |
| `run_manifest.md` | Human-readable run summary that links the preflight decision, resume semantics, authoritative files, and comparison outputs |
| `loop_assessment.json` | Workflow assessment report containing reuse eligibility, safe workflow loops, missing artifacts, and recommended next actions |
| `network_gate_validation.json` | Pre-execution validation results for stage-isolated network gate, including artifact fingerprints and validation status |
| `network_gate_execution.json` | Execution metadata for stage-isolated network gate runs, including timing, parity status, and resource usage |

## Workflow Notes

- `source/slavv/apps/parity_cli.py` is the canonical parity CLI implementation,
  and `dev/scripts/cli/compare_matlab_python.py` remains a thin wrapper.
- The parity CLI should treat this staged layout as the default organization
  for generated runs.
- `slavv import-matlab` imports a MATLAB batch into checkpoint-compatible
  artifacts that can then be consumed by parity reruns from the `edges` or
  `network` stage.
- Full-comparison reruns now default to safe reuse when
  `99_Metadata/matlab_status.json` reports a completed reusable batch
  (`matlab_batch_complete=true` / `complete-noop`):
  - if reusable Python outputs already exist, orchestration performs
    analysis-only comparison and skips MATLAB launch
  - if Python outputs are missing or incomplete, orchestration reuses the
    existing imported-MATLAB bootstrap flow (`--skip-matlab` behavior) and
    continues Python from the requested parity stage
  - `run_snapshot.json` and `run_manifest.md` record that MATLAB launch was
    skipped due to a completed reusable batch, including the selected reuse
    mode (`analysis-only` or `python-rerun`)
- Low-level standalone comparison helpers may still emit analysis artifacts at
  the caller-selected root when invoked directly. Prefer the staged layout when
  building durable comparison outputs for inspection.

## Chapter-Specific Workflow Notes

The active chapter docs explain the workflow details that sit on top of this
layout contract:

- [Shared Neighborhood Claim Alignment](../chapters/shared-neighborhood-claim-alignment/README.md)
- [Shared Candidate Generation Alignment](../chapters/shared-candidate-generation/README.md)
- [Imported-MATLAB Parity Report](../chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md)

Those chapters cover reuse summaries, diagnostics, proof artifacts, and other
workflow-specific outputs that still write into the staged folders above.

## Repository Guidance

- Treat generated outputs under `comparisons/`, `comparison_output*/`, ad-hoc
  temp folders, and cache directories as disposable artifacts rather than
  source inputs for code changes.
- Keep documentation and code references pointed at the staged layout, not at
  one-off local output paths.
- If a workflow produces a new durable output convention, update this document
  and the relevant tests before relying on it.

## Current Work

Chapter-specific parity status now lives in
[`docs/chapters/shared-candidate-generation/README.md`](../chapters/shared-candidate-generation/README.md).
Keep this document focused on the run-layout contract. For translation
semantics and boundary rules, use
[`docs/reference/MATLAB_TRANSLATION_GUIDE.md`](./MATLAB_TRANSLATION_GUIDE.md).


