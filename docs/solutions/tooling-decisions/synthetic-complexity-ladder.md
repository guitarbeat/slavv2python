---
title: Synthetic complexity ladder until first MATLAB↔Python divergence
date: 2026-08-14
category: tooling-decisions
module: analytics/parity
problem_type: tooling_decision
component: tooling
severity: medium
applies_when:
  - "Operators want a bounded fake-volume ladder to pressure-test toys-always-match after the 32³ Y-junction exact dual-run"
  - Need stop-at-first-mismatch semantics on vertices, edges, or strands without launching Certification writers
  - Interpreting ladder_report.json outcomes without promoting results to ONE TRUTH
root_cause: missing_tooling
resolution_type: tooling_addition
related_components:
  - development_workflow
  - testing_framework
  - documentation
tags:
  - synthetic-complexity-ladder
  - parity-falsification
  - matlab-python-dual-run
  - stop-at-first-mismatch
  - not-certification
  - y-junction
  - soft-cap
---

# Synthetic complexity ladder until first MATLAB↔Python divergence

## Context

A single matching 32³ Y-junction dual-run (`workspace/experiments/tiny_synthetic_matlab_python_diff/`) already showed exact agreement after index alignment. That one toy success can inflate the intuition that “synthetic always matches,” which weakens trust in the full-volume residual story and makes operators reach for expensive crop or canonical writers too early.

(session history) Live skepticism of the ONE TRUTH “one-strand / claimed-energy ranking residual” framing drove that tiny dual-run first: generate a fresh 32³ Y-junction TIFF, run MATLAB Vectorization-Public and Python exact-route, then compare curated vertices, spatial edge pairs, and strand count. After 0-based vs 1-based alignment the baseline matched (3 vertices / 2 edges / 1 strand). The harness pattern (`run_tiny_synthetic_diff.py` + MATLAB driver) was then generalized into a progressive ladder via brainstorm → plan `docs/plans/2026-08-14-002-feat-synthetic-complexity-ladder-plan.md`.

The **synthetic complexity ladder** answers a narrower question: under the same MATLAB↔Python dual-run compare pattern as the tiny experiment, escalate a **fixed**, hand-defined set of fake volumes until the **first real mismatch** on vertices, edges, or strands — or until a soft size/time cap / last fixed rung if they still agree. It is a **falsification / early-break probe**, not Certification, not Phase 1 Closure, and not evidence that the ADR 0013 Claimed Trace Energy production fix is done. Soft-cap full match on every rung is an informative negative result for the ladder hypothesis only; it does not prove the full-volume residual is ranking-only, and it must never update ONE TRUTH or claim-run roots.

Implementation on branch `feat/synthetic-complexity-ladder` (PR #108 open / unmerged as of this writing) maps to three units: U1 named geometries in `slavv_python/utils/synthetic.py`, U2 strict compare in `slavv_python/analytics/parity/probes/synthetic_dual_run_compare.py`, U3 orchestrator `scripts/run_synthetic_complexity_ladder.py` plus report helpers in `synthetic_ladder_report.py`. Unit coverage for report/stop orchestration lives in `tests/unit/analytics/parity/test_synthetic_ladder_report.py` (eight unit tests).

## Guidance

### What it is

A short fixed ladder of four named synthetic TIFF geometries, run in escalation order:

1. `y_junction_32` — baseline matching Y-junction (~32³)
2. `double_junction_32` — topology step: second opposite-side branch (~32³)
3. `asymmetric_y_48` — geometry asymmetry: offset junction / unequal radii (~48³)
4. `y_junction_64` — size step in the same Y-family (~64³ soft-cap size rung)

Rung ids and builders are registered in `slavv_python/utils/synthetic.py` (`LADDER_RUNG_IDS`; `generate_ladder_rung_volume`). Max dimension per rung for soft-size checks is `LADDER_RUNG_MAX_DIM`. There is **no** open-ended or search-based volume generator.

Each rung dual-runs MATLAB Vectorization-Public (parameterized driver `scripts/vectorize_ladder_rung.m`) and Python `SlavvPipeline` with shared params mirroring the tiny experiment, then applies a **strict** first-break surface: curated vertex spatial keys → spatial undirected edge pairs → strand counts (`first_break_surface` in `synthetic_dual_run_compare.py`). Graded tiny-script residual bands are **not** the stop predicate.

### How to run

From the repo root (PowerShell):

```powershell
.\.venv\Scripts\python.exe scripts\run_synthetic_complexity_ladder.py
```

Useful flags:

| Flag | Behavior |
|------|----------|
| `--rung <id>` | Run only one named rung (smoke / single-step); choices from `LADDER_RUNG_IDS` |
| `--skip-matlab` | Reuse latest per-rung MATLAB batch dir (runtime under the experiment tree) instead of launching MATLAB |
| `--reuse-python` | Reuse existing per-rung Python exact_run checkpoints (runtime under the experiment tree) |
| `--soft-time-sec` | Soft wall-clock budget per side before refusing the next rung (default 180) |
| `--soft-size-max-dim` | Refuse starting a rung whose max dim exceeds this (default 64) |

### Stop-at-first-mismatch and outcome semantics

Orchestration (`run_ladder` in `scripts/run_synthetic_complexity_ladder.py`):

1. For each planned rung, optionally apply soft-cap **before** starting it (`soft_cap_blocks_next_rung` in `synthetic_ladder_report.py`): size if next max dim > policy; time if either prior side’s wall_sec exceeded the budget (null wall under reuse skips that side).
2. Run dual-run; status becomes `match`, `first_break`, `inconclusive`, or `failed`.
3. On `first_break`, assemble report with `outcome=first_break` and halt — later rungs are not executed.
4. On `inconclusive` / `failed` (MATLAB unavailable, non-zero, non-comparable artifacts), stop with that outcome — never claim match or soft-cap full match.
5. If all executed rungs match through the fixed list, `outcome=soft_cap_full_match` with `soft_cap_reason=end_of_ladder` (or `size` / `time` when soft-cap blocked the next rung).

Report payload always includes `note` = `NON_CERTIFICATION_NOTE`: "Synthetic complexity ladder - NOT Certification / NOT Phase 1. Do not update ONE TRUTH or claim-run roots from this report." Script exit code is 0 only for `first_break` or `soft_cap_full_match`.

### Artifact layout

Under `workspace/experiments/synthetic_complexity_ladder/` (gitignored experiment tree; dirs appear only after a run):

- Top-level `ladder_report.json`
- Per-rung isolation: input TIFF, MATLAB batch output, and Python exact_run checkpoints under each rung id so skip/reuse cannot mix geometries

### Unit-tested pure helpers

Default CI does **not** run full MATLAB dual-run. Unit coverage covers strict stop ordering, report assembly / soft-cap / AE1–AE3 orchestration, and named rung shapes / determinism.

## Why This Matters

Cheap dual-run falsification sits **before** crop reselection and full Edges/Network writers. A first-break on a named synthetic rung is a successful falsification of “toys always match” for this ladder — useful for refining ranking or discovery hypotheses — but it is **not** a Certification pass/fail and must not be read as Phase 1 Closure.

Conversely, if early rungs still match, that only contradicts a blanket “not that close” claim **on those synthetic geometries**. It does not close Phase 1, does not replace evaluated ADR 0012 proofs on a claim run root, and does not imply AUDIT_REPORT “0-genuine” or E1–E10 portfolio greens are Certification standing. Soft-cap full match is a finished informative negative for the ladder, not proof that the full-volume residual is ranking-only.

(session history) Win criterion was deliberately “any first real mismatch,” not “must look like the ranking residual,” so the ladder stays a falsifier rather than a Certification proxy.

## When to Apply

- Iterating ranking or discovery hypotheses and you need a **cheap** MATLAB↔Python dual-run before launching crop or canonical writers
- You need an **ordered complexity walk** that stops at the first strict vertices / edges / strands break (or soft-cap / end-of-ladder)
- Smoke-checking dual-run wiring on the known baseline (`--rung y_junction_32`) after harness changes
- Reusing prior MATLAB batches / Python checkpoints (`--skip-matlab`, `--reuse-python`) while iterating compare or report logic

Do **not** use ladder outcomes to update ONE TRUTH, promote claim-run roots, declare ADR 0013 done, or treat audit/E-series portfolio greens as Phase 1 Certification.

## Examples

### Full ladder (operator)

```powershell
.\.venv\Scripts\python.exe scripts\run_synthetic_complexity_ladder.py
```

Writes `workspace/experiments/synthetic_complexity_ladder/ladder_report.json`.

### Smoke baseline rung only

```powershell
.\.venv\Scripts\python.exe scripts\run_synthetic_complexity_ladder.py --rung y_junction_32
```

### Reuse artifacts while iterating report logic

```powershell
.\.venv\Scripts\python.exe scripts\run_synthetic_complexity_ladder.py --skip-matlab --reuse-python
```

### Expected report shapes (from unit tests)

- **First break (AE1):** earlier rungs match, later rung edges mismatch → `outcome=first_break`, `first_break_surface=edges`, later rungs not present in `ladder_rungs`.
- **Soft-cap full match (AE2):** all four rungs match → `outcome=soft_cap_full_match`, `soft_cap_reason=end_of_ladder`, note remains non-Certification.
- **Inconclusive:** MATLAB unavailable → `outcome=inconclusive` (never a false match).
- **Strict compare order:** vertices → edges → strands; first differing surface wins.

## Related

- [Parity experiment hygiene](../best-practices/parity-experiment-hygiene.md) — cheap-first / falsify-before-writer process sibling; cross-link when using the synthetic first step as a durable dual-run operator path
- [Compare raw watershed candidates to raw, finals to finals](../parity/raw-vs-final-candidate-compare.md) — Certification residual diagnosis (claimed-energy ranking); contrast so ladder dual-run outcomes are not folded into ADR 0013 ship narrative
- Plan: [2026-08-14-002-feat-synthetic-complexity-ladder-plan.md](../../plans/2026-08-14-002-feat-synthetic-complexity-ladder-plan.md)
- Precursor harness: `workspace/experiments/tiny_synthetic_matlab_python_diff/`
