# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-06-25

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

All four crop-harness stages are resolved against oracle `180709_E_crop_M_v2` (`batch_260624-105705`). Phase 1's remaining work is the **full canonical `180709_E`** sequence.

- **Energy:** ✅ CERTIFIED (crop v2) under the [ADR 0011](../docs/adr/0011-energy-float-certification-policy.md) gate (Accepted) — discrete `scale_indices` strict-zero; `energy.energy` within `np.allclose`, max \|Δ\| ≈2×10⁻¹¹ (cross-library NumPy/MKL drift, [ADR 0010](../docs/adr/0010-random-component-parity-suite.md)).
- **Vertices:** ✅ CERTIFIED (crop v2) — positions+scales exact (13,706=13,706), energies within tolerance.
- **Edges:** ✅ certified per [ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md). Double-transpose orientation bug fixed; per-step math matches MATLAB; residual is emergent watershed order-sensitivity. Bar = voxel **ownership-map** (~63.5%) + per-edge trace tolerance. **Do not chase edge-pair overlap — it is misleading.**
- **Network:** ✅ topology EXACT (strands 10,722/10,722, bifurcations 5,601/5,601 on curated edges); geometry sub-voxel under trace tolerance ([ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md)).
- **Bar policy:** energy/vertices = strict zero + `np.allclose`; edges/network = ADR 0012 spatial bars (R1a in the spec).

## Operating sequence

1. Before any writer action, check `slavv jobs list` and run status on the target run root.
2. To re-verify a crop stage, use the **/prove-parity** skill (or `slavv parity prove-exact --stage <stage> --source-run-root <run> --dest-run-root <run> --oracle-root workspace/oracles/180709_E_crop_M_v2`). Interpret results with the per-stage bar above.
3. **Next milestone:** run the full canonical `180709_E` native pipeline through network, then `prove-exact-sequence` against the canonical oracle (`180709_E_full_v2`); each stage judged by its bar (R1/R1a).
4. MATLAB R2019a ground-truth harness for edge/network triage lives at `workspace/scratch/matlab_edge_instr/`.

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above.