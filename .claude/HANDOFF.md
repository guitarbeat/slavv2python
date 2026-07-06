# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-07-06

This is the single successor brief for the current exact-route effort. Do not use
dated agent passovers, PID snapshots, or parallel-work checklists as current
status.

## Canonical records

| Need | Source of truth |
|---|---|
| Active work and checkboxes | [docs/TODO.md](../docs/TODO.md) |
| Verified run status, proof evidence, and blockers | [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](../docs/plans/phase-1-exact-route-spec.md) |
| Edges/Network bar + closure policy | [ADR 0012 addendum](../docs/adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-06-phase-1-closure-bar-vs-strict-field-stretch) |
| Run commands and evidence format | [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md), [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md) |
| Repository and parity guardrails | [AGENTS.md](../AGENTS.md) |

## Current decision point

> **Single canonical status source:** [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) holds authoritative per-stage status. This section is the operator synthesis (2026-07-06).

**Dual bar (agreed 2026-07-06):**

- **Ship gate:** ADR 0012 per-stage `prove-exact` on **full** `180709_E` (`canonical_full_v5`).
- **Stretch (non-blocking):** strict-field + candidate-overlap KPI on refreshed **crop** (`crop_M_exact_v3`).

**Certified today (do not rerun):** Energy + Vertices on full `180709_E` (`canonical_full_v4` / `180709_E_full_v2`).

**Stale for closure claim:** Edges/Network on `canonical_full_v4` and `crop_M_exact` (pre–PR #103 checkpoints, 2026-07-04). `prove-exact-sequence` strict-field failures on those runs are **stretch signal**, not the Phase 1 closure gate.

**Next operator milestone:** execute the [operating sequence](#operating-sequence) below → per-stage ADR 0012 proof on `canonical_full_v5` → **Phase 1 closure** if green.

## Operating sequence

1. Before any writer: `slavv jobs list` + run-root status; no concurrent writers on the same dest root.
2. **Crop stretch baseline (`crop_M_exact_v3`):** preflight from `crop_M_exact` → rerun **edges only** on current `main` → log candidate-overlap KPI:
   ```powershell
   python scripts/watershed_candidate_gap_probe.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2
   ```
3. **Canonical closure run (`canonical_full_v5`):** preflight from `canonical_full_v4` (carry Energy/Vertices) → rerun **edges → network** on current `main`.
4. **Phase 1 closure proof** (ship gate only):
   ```powershell
   slavv parity prove-exact --stage edges `
     --source-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
     --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
     --oracle-root workspace/oracles/180709_E_full_v2

   slavv parity prove-exact --stage network `
     --source-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
     --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
     --oracle-root workspace/oracles/180709_E_full_v2
   ```
5. **If step 4 fails ADR 0012:** Phase 1 stays open — triage measurement (freshness, `(64,512,512)` shapes, oracle pairing, ownership probe) before watershed code changes. Iterate fixes on crop `v3`; do not claim closure.
6. **If step 4 passes:** declare Phase 1 closed; continue **strict-field stretch** on crop `v3` (overlap KPI daily, strict-field proof at milestones). Fix surface remains `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py` — selection/cleanup proven faithful.

Diagnostics: `workspace/scratch/edge_gap_split.py`, `edge_funnel_probe.py`, `scripts/watershed_candidate_gap_probe.py`; MATLAB harness `workspace/scratch/matlab_edge_instr/`.

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above.
