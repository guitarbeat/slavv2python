# MATLAB vs Python Parity Feedback-Loop Implementation Plan

This document replaces the old bottleneck checklist with a staged plan for
making parity work faster and more repeatable without treating long fresh
MATLAB-vs-Python runs as the default development loop.

## Goal And Constraint

- Exact parity is the target.
- MATLAB remains the reference implementation and should change minimally.
- Python is the side we iterate on for orchestration, reuse, diagnostics, and
  convergence work.
- Fresh full MATLAB-enabled comparison runs are too expensive to use after
  every Python-side change.
- The fastest path is to reduce unnecessary MATLAB reruns and reduce expensive
  deep comparisons when cheaper checks can answer the current question.
- Wrapper-level stability, reuse mechanics, and run-surface ergonomics are in
  scope. MATLAB algorithm changes are not the primary lever.

## Current Working Baseline

### Implemented now

- `workspace/scripts/cli/compare_matlab_python.py` supports `--validate-only`
  so output-root preflight can run without launching MATLAB or Python.
- `workspace/scripts/cli/compare_matlab_python.py` supports
  `--minimal-exports` so Python comparison runs can skip extra VMV/CASX/CSV
  export work.
- The staged comparison layout is now the canonical surface:
  `01_Input/`, `02_Output/`, `03_Analysis/`, and `99_Metadata/`.
- Standalone comparison already prefers checkpoint-backed Python results when
  checkpoints are available, then falls back to exported comparison JSON, then
  `network.json`.
- Full comparison orchestration already imports MATLAB `energy` and `vertices`
  into Python checkpoints and reruns Python from `edges` when those stages are
  available.
- Output-root preflight decisions, MATLAB rerun semantics, and MATLAB failure
  summaries already persist under `99_Metadata/`.

### Verified parity baseline

- Exact vertex parity is already established on the imported-MATLAB parity
  surface.
- Imported MATLAB `energy` plus imported MATLAB `vertices`, followed by a
  Python rerun from `edges`, is the correct current surface for convergence
  work.
- Candidate-endpoint coverage is already a useful first-pass diagnostic and
  should remain the first triage signal before final edge and strand diffs.
- Native Python-from-`energy` runs are still far from parity on the current
  canonical workload and should not be treated as the primary parity loop.

### Planned next

- Strict standalone source selection so comparison can choose one source
  explicitly instead of probing all fallbacks.
- `--resume-latest` run-root reuse for parity iterations.
- Shallow versus deep comparison control so full MATLAB parse is optional when
  only summary signals are needed.
- A lightweight MATLAB warm-up or health-check command before long live runs.

## Recommended Development Loops

Use the cheapest loop that can answer the current question.

### 1. Preflight loop

- Purpose: verify output-root health, free space, and launch readiness before a
  live MATLAB-enabled run.
- Use when: before any fresh MATLAB launch, before switching output roots, and
  before promoting a scratch path into a canonical run root.
- Reuses: only the selected output root and its current metadata state.
- Does not prove: algorithmic parity, imported-artifact validity, or anything
  about edge and strand convergence.
- Implemented now: `--validate-only`.

### 2. Analysis-only loop

- Purpose: reuse existing MATLAB and Python outputs to answer comparison
  questions without rerunning either pipeline.
- Use when: checking count deltas, parity-gate status, candidate coverage, or
  whether a previous run already contains the evidence needed for triage.
- Reuses: existing staged outputs under `01_Input/`, `02_Output/`, and
  `03_Analysis/`.
- Does not prove: that a fresh rerun would reproduce the same result on a new
  output root.
- Implemented now: standalone comparison already prefers checkpoints over
  exported comparison JSON and `network.json`.

### 3. Python parity loop

- Purpose: iterate on Python parity behavior without relaunching MATLAB.
- Use when: debugging edge and strand mismatches after a successful MATLAB
  batch already exists.
- Reuses: a successful MATLAB batch, imported MATLAB `energy`, imported MATLAB
  `vertices`, Python checkpoints, and staged metadata.
- Does not prove: full fresh-run confirmation against a new live MATLAB
  execution.
- Implemented now: orchestration already imports MATLAB `energy` and
  `vertices` and reruns Python from `edges` when those stages are present.
- Recommended default: this is the primary convergence loop until full
  fresh-run parity is close enough that fresh MATLAB launches are mainly
  milestone checks.

### 4. Full confirmation loop

- Purpose: confirm that the current Python state still matches MATLAB under a
  fresh live MATLAB-enabled comparison run.
- Use when: validating a milestone, updating the trusted canonical parity run,
  or checking whether accumulated Python-side changes survive a clean rerun.
- Reuses: normalized parameters, staged layout, and output-root preflight.
- Does not prove: that this should be the default loop for day-to-day Python
  debugging.
- Recommended use: reserve this for milestone confirmation rather than for each
  parity iteration.

## Phased Implementation Plan

### Phase 1: Cut wasted rerun time

- Add strict standalone source selection so comparison can choose
  `checkpoints-only`, `export-json-only`, or `network-json-only` instead of
  probing all fallbacks every time.
- Add `--resume-latest` so parity iterations can reuse the latest compatible
  run root instead of creating a new full run by default.
- Print explicit "reuse this output dir for the next parity rerun" guidance at
  the end of successful runs so developers naturally stay on the reuse path.
- Add a shallow versus deep comparison split so full MATLAB parse
  (`load_matlab_batch_results`) is optional when only summary counts,
  candidate-endpoint coverage, or parity-gate signals are needed.
- Add a lightweight MATLAB warm-up or health-check command for launch
  diagnostics before committing to a long live run.

### Phase 2: Make parity triage cheaper

- Treat imported MATLAB `energy` plus imported MATLAB `vertices` reruns as the
  default Python parity surface for edge and strand convergence work.
- Promote candidate-endpoint coverage and related candidate diagnostics to the
  first triage gate before final edge and strand diff inspection.
- Reduce duplicate filesystem work by sharing one inventory pass for manifest
  generation, size reporting, and file inventory generation.
- Cache preflight and repeated metadata inspections when the output root and
  relevant inputs have not changed.
- Add explicit doc guidance that native Python-from-`energy` runs are not the
  primary parity signal right now and should not block imported-energy parity
  work.

### Phase 3: Improve reliability and operator ergonomics

- Strengthen output-root guidance so fresh MATLAB runs default to local,
  non-synced storage instead of OneDrive or other fragile roots.
- Keep a clear separation between the trusted canonical parity run and
  disposable scratch reruns.
- Define a promotion path: run first on fast local storage, then keep or copy
  the finished run if it becomes the new reference.
- Add a short operator checklist for MATLAB failures: inspect `99_Metadata`,
  reuse the same output root when the metadata says it is valid, and avoid
  restarting from scratch unless the run surface is actually compromised.

## Success Criteria

- Most Python parity iterations avoid relaunching MATLAB.
- Most comparison iterations avoid deep MATLAB parsing unless detailed parity
  analysis is actually needed.
- The default developer loop becomes: reuse an existing MATLAB batch, import
  checkpoints, rerun Python from `edges`, run a cheap comparison, and only then
  schedule a fresh full run when a milestone needs confirmation.
- Fresh full MATLAB runs become milestone verification rather than the normal
  debugging loop.
- Output-root mistakes and environment failures are detected before expensive
  MATLAB launches whenever possible.

## Completed Work

- [x] Add `--validate-only` mode to
      `workspace/scripts/cli/compare_matlab_python.py` and
      `orchestrate_comparison()` so output-root preflight can run without
      launching pipelines.
- [x] Add `--minimal-exports` mode to reduce Python comparison export overhead
      by skipping VMV/CASX/CSV/JSON extras.
- [x] Persist output-root preflight decisions to `99_Metadata/output_preflight.json`.
- [x] Persist MATLAB rerun semantics and failure summaries to
      `99_Metadata/matlab_status.json` and
      `99_Metadata/matlab_failure_summary.json`.
- [x] Route imported-MATLAB parity reruns through checkpoint import and rerun
      Python from `edges` when MATLAB `energy` and `vertices` are available.
- [x] Prefer checkpoint-backed standalone comparison when Python checkpoints
      are available.

## Immediate Next Actions

- [ ] Implement strict standalone source selection so comparison can skip
      fallback probing when the caller already knows which Python result source
      is authoritative.
- [ ] Implement `--resume-latest` and pair it with explicit end-of-run reuse
      guidance so parity iterations naturally stay on the cheapest rerun path.
- [ ] Add a shallow versus deep comparison switch so candidate coverage,
      parity-gate signals, and summary counts can be checked without always
      loading full MATLAB batch results.
- [ ] Add a lightweight MATLAB warm-up or health-check command that validates
      launch viability before a long live comparison run.
- [ ] Add cached metadata and preflight inspection so repeated parity triage in
      the same output root does not redo avoidable work.
