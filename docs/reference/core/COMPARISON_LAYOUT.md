# Comparison Run Layout

[Up: Documentation Index](../../README.md)

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

## Run Root Naming

Use date-first names so roots sort naturally and are easy to scan.

- Preferred full timestamp form: `YYYYMMDD_HHMMSS_<label>`
  - example: `20260327_150656_clean_parity`
- Preferred date-only form: `YYYYMMDD_<label>`
  - example: `20260418_claim_ordering_trial`
- Avoid suffix-date form for new roots.
  - use `20260418_claim_ordering_trial`, not `claim_ordering_trial_20260418`

For one-off historical cleanup, rename only when it improves discoverability and
does not break active external references.

## Recommended Output Root

For live MATLAB-enabled comparisons, prefer a local non-synced drive with
comfortable free space. Good examples are:

- Windows: `D:\slavv_comparisons\20260413_release_verify`
- Windows (repo-local experiment archive): `C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\experiments\live-parity\runs\20260418_claim_ordering_trial`
- POSIX: `/tmp/slavv_comparisons/20260418_claim_ordering_trial`

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

Managed archive note:

- Runs under `slavv_comparisons/` are compacted after successful analysis.
- The retained Python surface is analysis-first: `network.json` when available,
  `python_comparison_parameters.json`, `stages/*/stage_manifest.json`,
  `stages/edges/candidate_audit.json`, and
  `stages/edges/candidate_lifecycle.json`.
- Heavy resumable payloads such as checkpoint `*.pkl`, `candidates.pkl`,
  VMV/CASX/CSV exports, and raw MATLAB batch bulk are pruned from the managed
  archive copy after the final comparison artifacts are written.

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
- `status.json` - explicit lifecycle metadata for managed archive runs
- `output_preflight.json`
- `matlab_status.json`
- `matlab_failure_summary.json` when MATLAB fails
- `artifact_cleanup.json` - managed archive cleanup profile, skip/apply status, and reclaimed size
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
| `status.json` | Explicit lifecycle metadata for managed archive runs, including `state`, `retention`, `quality_gate`, and optional supersession links |
| `output_preflight.json` | Authoritative preflight decision for the selected output root, including warnings, fatal errors, free-space estimates, and recommended action |
| `matlab_status.json` | Normalized MATLAB rerun semantics such as selected `batch_*` folder, resume mode, next stage, partial-artifact detection, and predicted rerun behavior |
| `matlab_failure_summary.json` | Concise failure summary and log-tail evidence persisted when MATLAB exits unsuccessfully |
| `artifact_cleanup.json` | Managed archive cleanup record describing whether compaction ran, the applied retention profile, and how many files/bytes were removed |
| `run_manifest.md` | Human-readable run summary that links the preflight decision, resume semantics, authoritative files, and comparison outputs |
| `loop_assessment.json` | Workflow assessment report containing reuse eligibility, safe workflow loops, missing artifacts, and recommended next actions |
| `network_gate_validation.json` | Pre-execution validation results for stage-isolated network gate, including artifact fingerprints and validation status |
| `network_gate_execution.json` | Execution metadata for stage-isolated network gate runs, including timing, parity status, and resource usage |

## Workflow Notes

## Optional Organization Layer

Managed comparison archives may add an organization layer above the staged run
root without changing the staged folders inside each run:

| Path | Purpose |
| --- | --- |
| `experiments/<experiment_slug>/runs/<run_root>/` | Optional grouping for related runs; the run root under `runs/` still owns the canonical staged folders |
| `experiments/<experiment_slug>/index.json` | Machine-readable summary of grouped runs for quick comparisons |
| `pointers/latest_completed.txt` | Repo-relative path to the latest completed managed run |
| `pointers/canonical_acceptance.txt` | Repo-relative path to the preferred acceptance/reference run |
| `pointers/best_saved_batch.txt` | Repo-relative path to the preferred reusable saved-batch run |

Notes:

- This grouping layer is optional and backward compatible. Legacy top-level run
  roots remain valid, and the staged layout above remains authoritative.
- A current repo-local example is
  `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`.
- Pointer files store exactly one path relative to the comparison root, not an
  absolute filesystem path.
- When a run is written under a managed archive root, the workflow now
  automatically refreshes `99_Metadata/status.json`, the containing
  `experiments/<slug>/index.json`, and the pointer files under `pointers/`.
- Aggregate roots that contain `run_*` children are treated as containers only.
  Each `run_*` child is the authoritative managed run; the container itself is
  used only for reporting and migration rollups.

## Lifecycle Status Metadata

When a managed archive workflow or comparison-layout maintenance pass touches a
run, it persists `99_Metadata/status.json` with this contract:

```json
{
  "state": "completed",
  "retention": "keep",
  "quality_gate": "partial",
  "supersedes": null,
  "superseded_by": null,
  "notes": "Optional human note"
}
```

Allowed values:

- `state`: `completed`, `failed`, `incomplete`, `superseded`, `archived`
- `retention`: `keep`, `eligible_for_cleanup`, `archive`
- `quality_gate`: `pass`, `fail`, `partial`, `unknown`

Readers prefer explicit `status.json` metadata over `run_snapshot.json` or
artifact heuristics when both exist. Runs targeted by a pointer file should be
kept with `retention=keep`.

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
- Managed archive runs under `slavv_comparisons/` are preserved for
  analysis/reference by default, not as resumable scratch runs. Once a managed
  run has a completed comparison report, archive maintenance compacts the run to
  the retained analysis surface and records the result in
  `99_Metadata/artifact_cleanup.json`.

## Chapter-Specific Workflow Notes

The active chapter docs explain the workflow details that sit on top of this
layout contract:

- [Neighborhood Claim Alignment](../../chapters/neighborhood-claim-alignment/README.md)
- [Candidate Generation Handoff](../../chapters/candidate-generation-handoff/README.md)
- [Imported-MATLAB Parity Closeout](../../chapters/imported-matlab-parity-closeout/parity_closeout.md)

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

## Cleanup And Retention

Use this policy when consolidating comparison outputs for faster reference and
lower noise:

- Keep roots with completed `03_Analysis/comparison_report.json` and
  `03_Analysis/summary.txt`.
- Keep roots that are canonical acceptance surfaces, release-audit references,
  or active chapter evidence.
- Prefer compaction over deletion for completed managed archive runs: keep
  `03_Analysis/**`, `99_Metadata/**`, `02_Output/python_results/network.json`
  when present, `python_comparison_parameters.json`, stage manifests, and the
  edge candidate audit/lifecycle artifacts.
- Remove roots explicitly marked failed in `99_Metadata/run_snapshot.json` when
  a completed replacement run exists.
- Remove empty folders after artifact moves.
- Do not delete canonical run roots referenced by active docs without first
  updating those links.

## Quick Validation Checklist

After any manual normalization pass:

1. Verify analysis artifacts are under `03_Analysis/`, not at run-root level.
2. Verify MATLAB batch folders live under `01_Input/matlab_results/`.
3. Verify status and manifest files live under `99_Metadata/`.
4. Verify run-root naming matches date-first convention.
5. Verify chapter/docs links still resolve to existing run roots.

## Current Work

Chapter-specific parity status now lives in
[`docs/chapters/neighborhood-claim-alignment/README.md`](../../chapters/neighborhood-claim-alignment/README.md).
Keep this document focused on the run-layout contract. For translation
semantics and boundary rules, use
[`docs/reference/core/MATLAB_TRANSLATION_GUIDE.md`](MATLAB_TRANSLATION_GUIDE.md).
