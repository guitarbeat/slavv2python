# ADR 0009: Parity Pre-Gate Tiers

## Status
Accepted

## Context
Phase 1 exact-route **Certification** on full `180709_E` requires multi-day pipeline runs and strict `prove-exact` gates. Iterating only on the canonical volume is slow, and Python-generated CI volumes do not have preserved MATLAB truth. The team needed a reproducible faster loop without weakening the certification claim or reusing the full-volume oracle on unrelated subvolumes.

## Decision
Adopt a three-tier **Parity Pre-Gate** before (and in parallel with) canonical certification:

1. **Synthetic fixture** — richer than a single straight tube (e.g. junction topology) for CI pipeline smoke only. No oracle, no `prove-exact` certification claim.
2. **Crop harness** — fixed center ROI **64×256×256** (Z×Y×X) cut from `180709_E.tif`, ids **`180709_E_crop_M`**, with a **new Oracle** produced by running MATLAB vectorization on that crop TIFF and `promote-oracle`. Same strict zero missing/extra sequential `prove-exact` bar as canonical cert, but success on crop does **not** satisfy Phase 1 **Certification** on full `180709_E`.
3. **Canonical volume** — full `180709_E` with oracle `180709_E_batch_190910-103039` remains the only surface for the Phase 1 exact-route certification milestone.

Full canonical runs may proceed in parallel with crop harness work; crop is for iteration, not a hard blocker on restarting canonical energy.

## Considered Options
- **Certify on synthetic or subset oracle from full E** — rejected: no authoritative MATLAB truth for synthetic; spatial subset of full-E `.mat` files is not valid unless MATLAB ran on the identical crop.
- **Pause all canonical work until crop passes** — rejected: team chose parallel execution to avoid idle wall-clock on long canonical runs.
- **Relaxed match-rate bar on crop** — rejected: crop must use the same strict zero bar or it fails to predict full-volume closure.

## Consequences
- One-time MATLAB vectorization cost per crop definition (`180709_E_crop_M`).
- Workflow and ROI bounds live in `docs/reference/workflow/PARITY_PRE_GATE.md`.
- Glossary terms in `GEMINI.md`: Parity Pre-Gate, Synthetic Fixture Volume, Crop Harness Volume, Canonical Volume.
