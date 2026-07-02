# ADR 0009: Parity Pre-Gate Tiers

## Status
Accepted

## Context
Phase 1 exact-route **Certification** on full `180709_E` requires multi-day pipeline runs and strict `prove-exact` gates. Iterating only on the canonical volume is slow, and Python-generated CI volumes do not have preserved MATLAB truth. The team needed a reproducible faster loop without weakening the certification claim or reusing the full-volume oracle on unrelated subvolumes.

## Decision
Adopt a three-tier **Parity Pre-Gate** before (and in parallel with) canonical certification:

1. **Synthetic fixture** — richer than a single straight tube (e.g. junction topology) for CI pipeline smoke only. No oracle, no `prove-exact` certification claim.
2. **Crop harness** — fixed center ROI **64×256×256** (Z×Y×X) cut from `180709_E.tif`, ids **`180709_E_crop_M`** (regenerated to `180709_E_crop_M_v2`; v1 stale on the scale plane), with a **new Oracle** produced by running MATLAB vectorization on that crop TIFF and `promote-oracle`. Same strict zero missing/extra sequential `prove-exact` bar as canonical cert, but success on crop does **not** satisfy Phase 1 **Certification** on full `180709_E`.
3. **Canonical volume** — full `180709_E` with oracle `180709_E_full_v2` remains the only surface for the Phase 1 exact-route certification milestone.

Full canonical runs may proceed in parallel with crop harness work; crop is for iteration, not a hard blocker on restarting canonical energy.

## Considered Options
- **Certify on synthetic or subset oracle from full E** — rejected: no authoritative MATLAB truth for synthetic; spatial subset of full-E `.mat` files is not valid unless MATLAB ran on the identical crop.
- **Pause all canonical work until crop passes** — rejected: team chose parallel execution to avoid idle wall-clock on long canonical runs.
- **Relaxed match-rate bar on crop** — rejected: crop must use the same strict zero bar or it fails to predict full-volume closure.

## Consequences
- One-time MATLAB vectorization cost per crop definition (`180709_E_crop_M`).
- Workflow and ROI bounds live in `docs/reference/workflow/PARITY_PRE_GATE.md`.
- Glossary terms in [AGENTS.md](../../AGENTS.md): Parity Pre-Gate, Synthetic Fixture Volume, Crop Harness Volume, Canonical Volume.
- Complementary fast component loop (no cert claim): [ADR 0010](0010-random-component-parity-suite.md), [PARITY_RANDOM_COMPONENT_SUITE.md](../reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md).

## Amendment (2026-07-01)

The canonical certification oracle was regenerated to `180709_E_full_v2` (fresh lattice-6000 MATLAB batch `batch_260626-125646`) and the crop oracle to `180709_E_crop_M_v2` (`batch_260624-105705`). The original 2019-vintage `180709_E_batch_190910-103039` batch predated the lattice-6000 / IFFT fixes and carried a +1 (0-based) `scale_indices` scale-plane drift, so it is retained as historical/legacy only and is no longer the Phase 1 claim surface. The three-tier structure of this ADR is unchanged. See docs/reference/core/EXACT_PROOF_FINDINGS.md.
