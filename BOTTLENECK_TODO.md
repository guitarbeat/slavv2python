# MATLAB vs Python Parity Workflow Plan

This file is the active parity backlog and investigation log.

Use [AGENTS.md](AGENTS.md) for the canonical repository workflow commands and
[docs/README.md](docs/README.md) for the maintained reading path.

Use this file when you want to know:

- which loop to run next
- which comparison features already exist
- which workflow improvements are still worth building

Use [docs/chapters/imported-matlab-parity/PARITY_HUB.md](docs/chapters/imported-matlab-parity/PARITY_HUB.md) for the fastest re-entry,
[docs/chapters/imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md](docs/chapters/imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md) for
evidence, and
[docs/chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md](docs/chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md)
for the remaining edge-generation work.

## Rapid Recall

- Exact parity is still open at `edges`.
- Already solved on the imported-MATLAB parity surface:
  - vertices
  - stage-isolated `network` when exact MATLAB `edges` are imported and Python
    reruns from `network`
- Default development loop:
  - reuse MATLAB batch
  - import MATLAB `energy` and `vertices`
  - rerun Python from `edges`
- Default downstream gate:
  - reuse MATLAB batch
  - import MATLAB `energy`, `vertices`, and `edges`
  - rerun Python from `network`
- Current best diagnosis:
  - the remaining gap is systematic
  - the active problem surface is frontier candidate generation and local
    partner selection
  - generic downstream network assembly is no longer the primary suspect

## Goal And Constraint

- Exact parity is the target.
- MATLAB remains the control and should change minimally.
- Python is where we iterate on orchestration, reuse, diagnostics, and
  convergence.
- Fresh full MATLAB-enabled runs are too expensive to use as the default loop.
- The main workflow goal is to shorten the feedback loop before exact full-run
  confirmation.

## Current Workflow Surface

### 1. Preflight loop

- Purpose: verify output-root health, free space, and launch readiness.
- Use when: before any fresh MATLAB launch or when changing output roots.
- Implemented now: `--validate-only`.
- Does not prove: anything about parity.

### 2. Analysis-only loop

- Purpose: compare existing artifacts without rerunning MATLAB or Python.
- Use when: checking counts, parity-gate status, candidate coverage, or source
  provenance from an existing run.
- Implemented now:
  - `--standalone-matlab-dir`
  - `--standalone-python-dir`
  - `--python-result-source`
  - `--comparison-depth shallow|deep`
- Does not prove: that a fresh rerun would reproduce the same result.

### 3. Default imported-MATLAB edge loop

- Purpose: iterate on edge-generation parity without relaunching MATLAB.
- Use when: working on `tracing.py` or any code that changes edge candidates or
  chosen edges.
- Reuses:
  - staged MATLAB batch
  - imported MATLAB `energy`
  - imported MATLAB `vertices`
  - Python checkpoints
- Implemented now:
  - `--skip-matlab`
  - `--resume-latest`
  - `--python-parity-rerun-from edges`
- Does not prove: fresh live MATLAB confirmation.

### 4. Stage-isolated network gate

- Purpose: prove whether a parity regression is really in `edges` rather than
  downstream network assembly.
- Use when: validating that Python `network` still converges when fed exact
  MATLAB `edges`.
- Reuses:
  - staged MATLAB batch
  - imported MATLAB `energy`
  - imported MATLAB `vertices`
  - imported MATLAB `edges`
- Implemented now:
  - `--skip-matlab`
  - `--resume-latest`
  - `--python-parity-rerun-from network`
  - forced parity-mode network assembly
- Expected result on the current parity surface: exact vertices, exact edges,
  and exact strands.

### 5. Full confirmation loop

- Purpose: confirm the current Python state against a fresh live MATLAB run.
- Use when: validating a milestone or promoting a new canonical parity run.
- Reuses: staged layout, normalized params, and output-root preflight.
- Recommended use: milestone confirmation, not day-to-day debugging.

## Implemented Workflow Features

- `--validate-only` output-root preflight
- `--minimal-exports` for lighter Python comparison runs
- `--resume-latest` with provenance and params compatibility checks
- loop-specific staged-artifact checks for reuse decisions
- `--comparison-depth shallow|deep`
- analysis-only standalone comparison with explicit Python result-source choice
- explicit reuse guidance after successful runs
- persisted `99_Metadata` preflight and MATLAB-status reports
- shared filesystem inventory pass for manifest and size reporting
- imported-MATLAB reruns from `edges`
- stage-isolated imported-MATLAB reruns from `network`
- comparison-mode Python execution now forces `comparison_exact_network=True`
- normalized comparison params now persist both:
  - `comparison_exact_network`
  - `python_parity_rerun_from`

## Open Workflow And Tooling Work

- Add a lightweight MATLAB warm-up or health-check command before long live
  runs.
- Cache repeated metadata and preflight inspections when the output root and
  inputs have not changed.
- Extend `--resume-latest` artifact checks so reuse decisions can distinguish
  more loop-specific surfaces more explicitly.
- Improve CLI messaging so runs clearly say:
  - safe to reuse
  - safe to analyze only
  - requires fresh MATLAB run
- Keep generated summaries focused on the earliest useful divergence signal
  rather than repeating downstream noise.

## Success Criteria

- Most parity iterations avoid relaunching MATLAB.
- Most comparison iterations avoid deep MATLAB parsing unless detailed analysis
  is actually needed.
- The default developer loop is:
  - reuse existing MATLAB batch
  - import checkpoints
  - rerun Python from `edges`
  - run cheap comparison
  - use the stage-isolated `network` gate when needed
  - use a fresh full run only for milestone confirmation
- Output-root mistakes and environment failures are caught before expensive
  MATLAB launches whenever possible.

## Completed Work

- [x] `--validate-only`
- [x] `--minimal-exports`
- [x] `--resume-latest`
- [x] shallow versus deep comparison control
- [x] explicit standalone Python result-source selection
- [x] persisted preflight and MATLAB-status metadata
- [x] imported-MATLAB reruns from `edges`
- [x] stage-isolated imported-MATLAB reruns from `network`
- [x] reuse guidance after successful runs
- [x] provenance-sensitive and params-sensitive reuse gating
- [x] loop-specific artifact gating for reuse
- [x] shared inventory pass for manifest and size reporting
- [x] parity-mode comparison params persisted in replay-safe form

## Immediate Next Actions

- [ ] Add a MATLAB warm-up or health-check command.
- [ ] Add cached metadata and preflight inspection for repeated reuse loops.
- [ ] Improve CLI summaries for reuse eligibility.
- [ ] Keep the stage-isolated `network` gate cheap and reliable while edge work
      continues.
- [ ] Continue using the shared-vertex diagnostics to drive the next
      `tracing.py` iteration.

## Related Docs

- [docs/chapters/imported-matlab-parity/PARITY_HUB.md](docs/chapters/imported-matlab-parity/PARITY_HUB.md)
- [docs/chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md](docs/chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md)
- [docs/chapters/imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md](docs/chapters/imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md)
- [docs/reference/COMPARISON_LAYOUT.md](docs/reference/COMPARISON_LAYOUT.md)
- [workspace/reports/stage_isolated_network_parity_2026-04-07.md](workspace/reports/stage_isolated_network_parity_2026-04-07.md)
