# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-06-24

This is the single successor brief for the current exact-route effort. Do not use
dated agent passovers, PID snapshots, or parallel-work checklists as current
status.

## Canonical records

| Need | Source of truth |
|---|---|
| Active work and checkboxes | [docs/TODO.md](../docs/TODO.md) |
| Verified run status, proof evidence, and blockers | [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](../docs/plans/phase-1-exact-route-spec.md) |
| Run commands and evidence format | [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md), [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md) |
| Repository and parity guardrails | [AGENTS.md](../AGENTS.md) |

## Current decision point

Phase 1 remains blocked at the crop-harness **strict** Energy gate (`prove-exact`).

- **Oracle:** use `workspace/oracles/180709_E_crop_M_v2` (`batch_260624-105705`, lattice-6000). v1 is stale on `scale_indices`.
- **Writer:** job `75188cc2` completed; `inspect-energy-evidence` **valid** on `crop_M_exact`.
- **Strict proof:** `prove-exact --stage energy` vs v2 → **FAIL** on `energy.energy` only (`scale_indices` **0**).
- **ULP triage:** 3,810,126 scale-agreeing strict mismatches; median **4 ULP**, p90 **13 ULP**, max \|Δ\| **≈2×10⁻¹¹**; writer persistence ruled out. Cross-library NumPy/MKL drift ([ADR 0010](../docs/adr/0010-random-component-parity-suite.md)).
- **Advisory loop:** `slavv parity prove-energy-ulp` (strict scales + `--max-ulps`, default 8) — **not** certification.
- **Policy open:** [TODO § Energy certification policy](../docs/TODO.md) — strict `np.equal` vs documented ULP tolerance for Phase 1 claim.
- **Downstream frozen:** Vertices → Edges → Network until strict Energy passes.

## Operating sequence

1. Before any writer action, check `slavv jobs list` and run status on `crop_M_exact`.
2. Strict gate: `prove-exact --stage energy --oracle-root workspace/oracles/180709_E_crop_M_v2`.
3. Advisory ULP gate: `prove-energy-ulp --max-ulps 8` (same roots) for developer signal only.
4. After strict Energy strict-zero, refresh Crop Vertices → Edges → Network; `prove-exact-sequence` vs v2.
5. Only after crop sequence passes, resume full canonical `180709_E` sequence.

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above.