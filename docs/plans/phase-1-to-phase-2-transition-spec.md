---
title: "Phase 1 to Phase 2 transition"
type: spec
status: draft
date: 2026-07-13
topic: parity-closure-to-optimization-handoff
---

# Phase 1 to Phase 2 transition

**Purpose:** define the controlled handoff from exact-route certification to
optimization work. This spec prevents broad Phase 2 refactors from starting
before the full-volume Network ADR 0012 ship gate is green and the canonical
baseline is frozen.

**Live status:** [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk)  
**Operator brief:** [HANDOFF](../../.claude/HANDOFF.md)  
**Phase 1 spec:** [phase-1-exact-route-spec.md](phase-1-exact-route-spec.md)  
**Phase 2 draft:** [phase-2-optimization-spec.md](phase-2-optimization-spec.md)

> **Do not freeze a preferred run ID in this spec.** Use the claim/closure root named in ONE TRUTH (or a **new successor** of that root). Historical names such as `canonical_full_v7` are audit lineage only.

---

## Entry criteria

Do not start this transition until all are true on a **fresh** canonical full-volume
run root (successor of the current claim surface in ONE TRUTH; never overwrite prior audits):

1. Energy passes the ADR 0011 bar on full `180709_E`.
2. Vertices pass the ADR 0011 bar on full `180709_E`.
3. Edges pass an **evaluated** ADR 0012 proof against `180709_E_full_v2`.
4. Network passes an **evaluated** ADR 0012 proof against `180709_E_full_v2`.
5. Proof evidence is recorded in
   [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).
6. The closure run root and oracle are not overwritten after proof.

## Exit criteria

The transition is complete when:

1. The closure run root is marked as the frozen Phase 1 baseline in findings.
2. Artifact hashes for Energy, Vertices, Edges, and Network are recorded or
   referenced through release evidence.
3. Figures and proposal/manuscript summaries are regenerated from the closure
   metrics.
4. Phase 2 optimization workstreams have explicit regression gates.
5. TODO no longer has open Phase 1 closure checkboxes except optional
   strict-field stretch tasks.

## Frozen baseline artifacts

Freeze these surfaces before optimization:

| Artifact | Owner | Notes |
|---|---|---|
| Canonical run root | `workspace/runs/oracle_180709_E/<closure_root>` | Must include Edges and Network checkpoints produced by current main. |
| Oracle | `workspace/oracles/180709_E_full_v2` | Must include ownership map and all required normalized artifacts. |
| Proof reports | `<closure_root>/03_Analysis/` | Per-stage evaluated proofs, not strict-field `prove-exact-sequence` as closure. |
| Release evidence | `<closure_root>/03_Analysis/release_evidence.json` | Hash bridge for run, params, oracle, commit, and artifacts. |
| Figures | `figures/` | Regenerated after closure metrics settle. |

## Allowed before baseline freeze

The following are allowed while Network remains red:

- Watershed generation / claiming-state fixes tied to crop probes.
- Diagnostic instrumentation for the first divergent strel.
- Crop-only fast-loop probes and targeted unit tests.
- Documentation updates that keep findings, handoff, TODO, and figures aligned.

## Forbidden before baseline freeze

Do not start these while Network ADR 0012 is red:

- Broad C-order / Fortran-order unwinding.
- Replacing MATLAB-faithful kernels with idiomatic NumPy/SciPy variants.
- GPU or distributed rewrites that perturb parity surfaces.
- Relaxing Network multiset equality without a new ADR.
- Claiming Phase 1 closure from Edges-only pass, crop-only pass, or
  `prove-exact-sequence` strict-field output.

## Phase 2 workstreams

After the baseline is frozen, Phase 2 may proceed in separate tracked specs or
implementation tickets:

1. **Performance profiling:** establish stage runtime and memory baselines on
   the frozen canonical run.
2. **Energy optimization:** tune exact-route parallelism and evaluate GPU/FFT
   backends behind regression gates.
3. **Edges/network optimization:** profile global watershed and network assembly
   without changing accepted topology bars.
4. **Paper-profile certification:** define volume, oracle, and bars for the
   public `paper` profile.
5. **Figure/report finalization:** regenerate proposal and methods figures from
   closure metrics.

## Regression gates

Every Phase 2 optimization must state which gate applies:

| Change type | Minimum gate |
|---|---|
| Refactor that should preserve exact route behavior | Existing focused unit tests + relevant per-stage proof on crop or full surface. |
| Performance optimization with intended bit preservation | Byte or strict discrete equality against frozen baseline where feasible. |
| Deliberate bit-perturbing optimization | New ADR or explicit topological tolerance gate before merge. |
| Public paper-profile work | Paper-profile proof plan and oracle policy before claims. |

## Documentation updates at transition

When Phase 1 closes:

1. Update [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md)
   top banner and active operations.
2. Re-synthesize [.claude/HANDOFF.md](../../.claude/HANDOFF.md).
3. Update [TODO.md](../TODO.md) checkboxes.
4. Update [ROADMAP.md](../ROADMAP.md) to move Phase 1 to complete.
5. Regenerate figures and update [figures/README.md](../../figures/README.md).
6. Update this spec's status from `draft` to `complete` after baseline freeze.
