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
- `workspace/scripts/cli/compare_matlab_python.py` supports
  `--resume-latest` so parity iterations can reuse the latest discovered run
  root instead of creating a fresh timestamped directory by default.
- Comparison orchestration supports `--comparison-depth shallow|deep` so
  summary-count and parity-gate checks can skip full MATLAB batch parsing when
  deep MATLAB artifacts are not needed.
- The staged comparison layout is now the canonical surface:
  `01_Input/`, `02_Output/`, `03_Analysis/`, and `99_Metadata/`.
- Standalone comparison already prefers checkpoint-backed Python results when
  checkpoints are available, then falls back to exported comparison JSON, then
  `network.json`.
- Standalone comparison internals already support explicit Python result-source
  selection (`auto`, `checkpoints-only`, `export-json-only`,
  `network-json-only`).
- `workspace/scripts/cli/compare_matlab_python.py` now exposes the
  analysis-only loop directly through `--standalone-matlab-dir`,
  `--standalone-python-dir`, and `--python-result-source`, so existing
  artifacts can be compared without relaunching MATLAB or Python.
- Full comparison orchestration already imports MATLAB `energy` and `vertices`
  into Python checkpoints and reruns Python from `edges` when those stages are
  available.
- Output-root preflight decisions, MATLAB rerun semantics, and MATLAB failure
  summaries already persist under `99_Metadata/`.
- Successful comparison runs already print explicit reuse guidance so the next
  parity rerun naturally stays on the same staged output root.
- `--resume-latest` now checks recorded input provenance before reusing a
  discovered run root and falls back to a fresh timestamped run root when the
  newest candidate is clearly for a different input or normalized parameter
  set.
- `--resume-latest` now also checks that the requested loop has the staged
  artifacts it actually depends on before reuse is allowed. For example,
  Python-only parity reruns now require reusable Python checkpoints instead of
  treating any recent run root as resumable.
- Manifest generation and run inspection now share one filesystem inventory
  pass for total-size reporting and typed file inventory generation.
- Generated summaries now recommend candidate-endpoint coverage as the first
  triage surface when missing or extra candidate endpoint pairs are present.

### Verified parity baseline

- Exact vertex parity is already established on the imported-MATLAB parity
  surface.
- Watershed join supplement parity established (removed ~1,000 invalid joins by tightening energy rejection).
- Imported MATLAB `energy` plus imported MATLAB `vertices`, followed by a
  Python rerun from `edges`, is the correct current surface for convergence
  work.
- Candidate-endpoint coverage is already a useful first-pass diagnostic and
  should remain the first triage signal before final edge and strand diffs.
- Native Python-from-`energy` runs are still far from parity on the current
  canonical workload and should not be treated as the primary parity loop.

### Reflection From Recent Iterations

- The highest-leverage wins have come from Python-side workflow control rather
  than algorithm edits: cheaper comparison entry points, shallower analysis
  modes, and clearer reuse guidance all landed cleanly without disturbing the
  MATLAB control.
- The analysis-only loop is more valuable once it is first-class at the CLI.
  Internal support alone was not enough; engineers need a direct command path
  to force checkpoint-backed, export-JSON, or `network.json` comparison.
- Reuse needs provenance, not just recency. The newest run root is not a safe
  default if it cannot prove that it was created for the same input and
  normalized parameter surface.
- Reuse also needs the right staged surface for the requested loop. Matching
  provenance is still not enough if the run root does not contain the
  checkpoints or batch artifacts that make reuse meaningful.
- Candidate-endpoint coverage remains the cheapest useful mismatch signal and
  is now surfaced directly in generated summaries, which should keep triage
  focused on the earliest divergence point.

### Planned next

- Add a lightweight MATLAB warm-up or health-check command before long live
  runs.
- Cache preflight and other repeated metadata inspections when the output root
  and relevant inputs have not changed.
- Extend `--resume-latest` compatibility checks to more loop-specific staged
  surfaces and operator hints beyond the first requested-artifact checks.

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
  exported comparison JSON and `network.json`, and the CLI exposes this reuse
  path directly through `--standalone-matlab-dir`,
  `--standalone-python-dir`, and `--python-result-source`.

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
- [x] Add `--resume-latest` so parity iterations can reuse the latest existing
      run root.
- [x] Add shallow versus deep comparison control so summary checks can skip
      full MATLAB batch parsing.
- [x] Expose strict standalone source selection as a public CLI option so the
      analysis-only loop can choose an authoritative Python result source.
- [x] Persist output-root preflight decisions to `99_Metadata/output_preflight.json`.
- [x] Persist MATLAB rerun semantics and failure summaries to
      `99_Metadata/matlab_status.json` and
      `99_Metadata/matlab_failure_summary.json`.
- [x] Route imported-MATLAB parity reruns through checkpoint import and rerun
      Python from `edges` when MATLAB `energy` and `vertices` are available.
- [x] Prefer checkpoint-backed standalone comparison when Python checkpoints
      are available.
- [x] Print explicit reuse-output-dir guidance after successful comparison
      runs.
- [x] Gate `--resume-latest` reuse on recorded input provenance and fall back
      to a fresh run root when the newest candidate is incompatible.
- [x] Extend `--resume-latest` compatibility checks to validate normalized
      comparison parameters as well as recorded input provenance.
- [x] Extend `--resume-latest` compatibility checks to validate required staged
      artifacts for the requested loop before reuse is allowed.
- [x] Share one filesystem inventory pass across manifest generation and run
      size reporting.
- [x] Surface candidate-endpoint coverage as the first triage recommendation in
      generated comparison summaries.

## Immediate Next Actions

- [ ] Add a lightweight MATLAB warm-up or health-check command that validates
      launch viability before a long live comparison run.
- [ ] Add cached metadata and preflight inspection so repeated parity triage in
      the same output root does not redo avoidable work.
- [ ] Extend `--resume-latest` artifact checks beyond the first loop-specific
      rules so reuse decisions can distinguish checkpoint-backed Python reruns,
      reusable MATLAB batch surfaces, and analysis-only output roots more
      explicitly.
- [ ] Add lightweight operator messaging that distinguishes "safe to reuse",
      "safe to analyze only", and "requires fresh MATLAB run" directly in CLI
      summaries.
