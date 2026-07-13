# SLAVV MATLAB→Python Parity — Project Update

**Date:** 2026-07-01  
**Scope:** Bit-exact reproduction of the MATLAB SLAVV vessel-extraction pipeline
(Energy → Vertices → Edges → Network) in Python, certified against a frozen
MATLAB R2019a oracle.

> ⚠️ **Point-in-time snapshot (2026-07-01).** Do **not** use this file as current
> status. Live per-stage status, residual Network ship gate, and active blockers:
> **[EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)** · operator
> brief: **[.claude/HANDOFF.md](../.claude/HANDOFF.md)** · tasks: **[TODO.md](TODO.md)**.
> As of 2026-07-08+, full-volume Energy/Vertices/Edges ADR 0012 are green on the
> closure path; **Network ADR 0012 remains the open Phase 1 ship gate**.

---

## Executive summary

The Python port now reproduces the MATLAB pipeline **stage by stage**, validated against
a fixed R2019a oracle rather than by inspection. **All four stages meet their certification
bars on the crop harness** (full-canonical proofs beyond Energy are still in progress — see
the canonical status source below), the last certified here being Energy on the full
canonical volume, whose final discrepancy was root-caused, fixed, and confirmed this period.

The headline result: a long-standing energy mismatch on the full volume — ~39,500 voxels
(0.24%) choosing a different vessel scale than MATLAB — was traced through a purpose-built
MATLAB ground-truth harness to a **single sub-microscopic numerical cause** (a ~1e-15
difference in one interpolation coordinate), and corrected with a bit-exact port of MATLAB's
`linspace`. The full-volume proof then passed with **zero scale mismatches across all
16,777,216 voxels** — the canonical Energy stage is certified.

---

## Certification status by stage

> **Canonical status source:** [docs/reference/core/EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
> is the single source of truth for current per-stage parity status; the table below is a
> point-in-time snapshot (2026-07-01).

Certification is **evidence-based**: each stage's Python output is compared field-by-field
to the MATLAB oracle. Discrete fields (scale indices, graph topology) must match exactly;
continuous floating-point fields are compared within a tight tolerance (ADR 0011), because
cross-library float arithmetic (NumPy/MKL vs MATLAB) differs at the ~1e-11 level even when
the logic is identical.

| Stage | Status | Evidence |
|---|---|---|
| **Energy** (crop) | ✅ **Certified** | Scale indices exact (0 mismatches); energy within 2e-11; passes the ADR 0011 gate. |
| **Energy** (full canonical) | ✅ **Certified** | Last discrepancy (octave-3/4 scale mismatches) traced to the upsample interpolation mesh and fixed (see below). Full-volume proof passes: **0 scale mismatches across all 16,777,216 voxels**; energy within tolerance (max \|Δ\| ≈ 2.4e-11, ADR 0011). |
| **Vertices** | ✅ **Certified** | Positions and discovery scales exact; energies sourced from the MATLAB record. |
| **Edges** | ✅ **Certified (ADR 0012)** | The watershed is a greedy shared-state flood-fill whose exact voxel claims are order-sensitive; certified on **voxel-ownership agreement + per-edge trace tolerance**, after fixing a grid-orientation bug. The per-step math matches MATLAB exactly. |
| **Network** | ✅ **Topology exact; geometry sub-voxel (ADR 0012)** | Strand and bifurcation topology reproduce MATLAB **100%** (10,722 strands, 5,601 bifurcations, 0 missing/extra). Residual is sub-voxel smoothing drift, certified under a trace tolerance. |

---

## This period's focus: the canonical Energy divergence

**Problem.** On the full 64×512×512 volume the energy field selected a different vessel
scale than MATLAB at 39,494 of 16.8M voxels (0.24%), concentrated in the heavily
downsampled scale "octaves" 3–4. The crop volume certified cleanly, so the bug only
appeared at full scale.

**Method — MATLAB as ground truth.** Because the recorded oracle stores only the final
result (not the per-scale intermediates needed to localize the bug), we built an
instrumented MATLAB R2019a harness that replays the energy computation at individual
voxels and dumps every intermediate field. A matching Python probe lets us compare the
two implementations step-by-step at the exact failing voxels. (Harness:
`workspace/scratch/matlab_energy_instr/`.)

**Findings — disciplined elimination.** Several plausible explanations were tested and
**ruled out** with evidence, not assumption:
- Not parallelism (single- vs multi-threaded energy is byte-identical).
- Not chunk-boundary seams (mismatches sit in chunk interiors).
- Not a Hessian/curvature sign error (the matched-filter energy matches MATLAB to ~1e-14
  wherever both implementations agree a voxel is a vessel).

**Root cause.** The divergence is the Python coarse→fine **upsampling mesh not bit-matching
MATLAB's `linspace`**. At a downsampled-octave cell boundary, a ~1e-15 difference in one
mesh coordinate floors the interpolation into the *adjacent* coarse cell — and when that
neighbor is an "invalid" (non-vessel) cell, the interpolated energy collapses, flipping the
chosen scale. A first heuristic fix (snapping near-integer coordinates) was **rejected after
verification**: MATLAB's `linspace` does not always land on integers, so snapping fixed some
voxels and broke others.

**Fix.** A bit-exact port of MATLAB R2019a `linspace` (integer-modulo phase, MATLAB's
multiply-then-divide step, forced endpoints). It reproduces MATLAB to **<1e-17** on both an
integer-landing voxel and a sub-integer-landing voxel, the previously-failing residual
voxels now select MATLAB's scale, and the full Python test suite is green (595 passed).

**Status.** ✅ **Certified.** The full canonical energy proof passed with **0 scale
mismatches** across all 16.8M voxels (`prove-exact --stage energy`, exit 0) — the mesh
divergence is fully resolved.

---

## Methodology notes (transferable)

- **Oracle-anchored certification.** Every claim of "matches MATLAB" is a proof artifact
  comparing to a frozen R2019a run, not a code review.
- **Ground-truth instrumentation over guessing.** The hard bugs (edges, energy) were each
  solved by instrumenting the *MATLAB* reference at the exact failing location and comparing
  intermediates — which repeatedly overturned plausible-but-wrong hypotheses.
- **Verify before declaring done.** Both the edge "size-reference" fix and the energy
  "grid-snap" fix were caught as wrong *before* shipping because they were checked against
  MATLAB ground truth rather than accepted on plausibility.

---

## Remaining work

1. **Full-volume Vertices / Edges / Network proofs** — extend the certified crop results to
   the full canonical volume now that Energy is certified end-to-end.
2. **Performance / optimization pass** — now that parity is locked, optimize the Python path
   (the exact-route energy is the runtime bottleneck at full scale — the certifying run took
   ~9 h, dominated by the full-resolution first octave).

---

*Generated for a PI status update. Full technical record:
`docs/reference/core/EXACT_PROOF_FINDINGS.md` and
`docs/solutions/parity/canonical-energy-high-octave-divergence.md`.*
