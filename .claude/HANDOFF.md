# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-07-04

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

> **Single canonical status source:** [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) holds authoritative, up-to-date per-stage parity status; the verdicts below are a synthesis snapshot (2026-07-04).

The full canonical `180709_E` sequence ran (`canonical_full_v4`, 2026-07-04): **Energy + Vertices are now CERTIFIED strict on the full 16.8M-voxel volume**; **Edges + Network FAIL strict-field**. A 2026-07-04 debug session localized the failure to the watershed **candidate-generation** step (below) — this is Phase 1's single remaining substantive blocker.

- **Energy:** ✅ CERTIFIED (crop v2 **and full `180709_E_full_v2`**) under the [ADR 0011](../docs/adr/0011-energy-float-certification-policy.md) gate — discrete `scale_indices` strict-zero (0 / 16,777,216 voxels on full); `energy.energy` within `np.allclose`, max \|Δ\| ≈2×10⁻¹¹ (cross-library NumPy/MKL drift, [ADR 0010](../docs/adr/0010-random-component-parity-suite.md)).
- **Vertices:** ✅ CERTIFIED (crop v2 + full) — positions+scales exact, energies within tolerance.
- **Edges:** 🟡 passes [ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md) spatial bars (ownership-map ~63.5%); ⛔ **FAILS strict-field** on crop (13,555 vs 15,511) and full (60,213 vs 69,500). **Root cause (2026-07-04): watershed candidate-*generation* adjacency gap — 43% of MATLAB's final edges are never Python candidates; only 916 of the crop gap is pruning.** Pruning steps match MATLAB source; Python wires vertices to different neighbors. Fix surface: `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py`. **Do not chase edge-pair overlap — it is misleading; do not re-audit selection/cleanup — it is faithful.**
- **Network:** 🟡 passes ADR 0012 on curated crop edges; ⛔ **FAILS strict-field on full** (39,623 vs 48,049 strands) **entirely downstream of the edge deficit** — no independent network bug.
- **Bar policy:** energy/vertices = strict zero + `np.allclose`; edges/network = ADR 0012 spatial bars (R1a in the spec). Strict-field closure of edges/network requires the watershed generation fix.

## Operating sequence

1. Before any writer action, check `slavv jobs list` and run status on the target run root.
2. To re-verify a crop stage, use the **/prove-parity** skill (or `slavv parity prove-exact --stage <stage> --source-run-root <run> --dest-run-root <run> --oracle-root workspace/oracles/180709_E_crop_M_v2`). Interpret results with the per-stage bar above.
3. **Next milestone:** close the **watershed candidate-generation adjacency gap** (Edges/Network strict-field). Iterate on the crop (~5 min edges rerun): instrument `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py` at basin-meeting/edge-recording points, compare recorded adjacency against MATLAB for a sample of the 6,726 missing pairs (hypotheses H1–H5 in the findings). Do **not** re-audit selection/cleanup (proven faithful) or Network (downstream). Re-run the full `canonical_full_v4` sequence only after the crop generation gap closes.
4. MATLAB R2019a ground-truth harness for edge/network triage lives at `workspace/scratch/matlab_edge_instr/`. Offline gap-split diagnostics: `workspace/scratch/edge_gap_split.py`, `edge_funnel_probe.py`.

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above.