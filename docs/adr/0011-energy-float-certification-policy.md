# ADR 0011: Energy Float Certification Policy

## Status
Accepted (2026-06-24)

## Context

Phase 1 exact-route [Certification](../plans/phase-1-exact-route-spec.md) requires sequential `prove-exact` with **strict set equality** on every compared field (spec R1). The crop harness (`180709_E_crop_M`) is the iteration surface before the canonical `180709_E` claim ([ADR 0009](0009-parity-pre-gate-tiers.md)).

As of 2026-06-24, against oracle **`180709_E_crop_M_v2`** (`batch_260624-105705`, lattice-`6000`):

| Field | `prove-exact --stage energy` | Notes |
| --- | --- | --- |
| `scale_indices` | **0** mismatches | Fresh MATLAB promotion resolved stale v1 (+1) drift |
| `energy` | **3,810,126** mismatches | Strict `np.equal` on `energy.energy` |
| `lumen_radius_microns` | **8** mismatches | Machine epsilon on 99-element vector |

Energy evidence (scale-agreeing voxels only):

- **384,178** voxels bit-identical (~9% of volume)
- Median **4 ULP**, p90 **13 ULP**, max \|Δ\| **1.99×10⁻¹¹**
- Writer persistence ruled out (stored `best_energy.npy` matches single-octave Python replay on sampled mismatches)
- Voxel probes: Python live replay agrees with MATLAB on scale winners; float drift is **NumPy vs MATLAB MKL** at matching scales

[ADR 0010](0010-random-component-parity-suite.md) already documents a **≥1 ULP** floor on `ifftn(..., 'symmetric')` with identical complex spectra. Full-volume crop evidence shows **accumulated** drift (median 4 ULP), not a single localized Python bug.

A diagnostic ULP probe exists: `slavv parity prove-energy-ulp` (strict `scale_indices`, configurable `--max-ulps`). As of the accepted decision below, **certification uses the `np.allclose` gate** wired into `prove-exact`; the ULP probe is telemetry only.

| `max_ulps` | Crop pass rate | Failures |
| --- | --- | --- |
| 8 (one-voxel probe default) | ~82% | ~755k voxels |
| 48 (p99 on mismatches) | **99.11%** | 37,174 (mostly denormal/near-zero energies; \|Δ\| still ≤2×10⁻¹¹) |

**Status (resolved 2026-06-24):** Energy certification policy accepted (below); crop Energy `prove-exact` now **PASSES**. Downstream Vertices/Edges/Network gates are unblocked.

## Decision — Accepted (2026-06-24): Option B, refined to `np.allclose`

The Phase 1 certification gate for the Energy stage is:

- **Discrete / topological fields strict:** `scale_indices` (and array shapes) compare with exact equality — achieved (0 mismatches on crop v2).
- **Continuous float fields by tolerance:** `energy.energy`, `lumen_radius_microns`, `energy_4d`, and any other floating-point field pass when within `np.allclose(rtol=1e-7, atol=1e-9)`. ULP figures are retained as **diagnostics only**.

**Refinement vs the original `max_ulps=48` recommendation.** A full-volume re-run showed pure ULP is the wrong metric: ULP distance explodes for near-zero energies, rejecting **36,074** scale-agreeing voxels at 48 ULP (max **72,343** ULP) whose absolute error was ≤ 2×10⁻¹¹. Absolute/relative tolerance is the correct scientific bar and passes 100%.

**Demonstrated (2026-06-24, oracle `180709_E_crop_M_v2`):** `slavv parity prove-exact --stage energy` → **PASS** (exit 0).
- `energy`: max |Δ| = **1.99×10⁻¹¹** (rel 9.3×10⁻¹²), pass_rate 1.0
- `lumen_radius_microns`: max |Δ| = **7.1×10⁻¹⁵** (rel 2.4×10⁻¹⁶)
- `scale_indices`: **0** mismatches

**Implementation:** `evaluate_energy_float_gate` (`use_allclose=True`, default) decides `energy.energy`; the generic comparator (`artifact_comparator._compare_value`) applies `np.allclose` to all other float fields and strict equality to integer/topological fields. Strict `np.equal` remains available via `--strict-floats` for regression. ULP probe semantics (`prove-energy-ulp`) are unchanged.

The original option analysis below is retained for the record.

### Option A — Strict bit-identical (status quo)

- **Gate:** `prove-exact` continues `np.array_equal` on `energy.energy` with zero tolerance.
- **Scale:** `scale_indices` remains strict integer equality (already achieved on crop v2).
- **Pros:** Matches spec R1 verbatim; no semantic drift in certification claim.
- **Cons:** Crop Energy gate likely **blocked indefinitely** without a MATLAB-linked FFT backend or other bit-identical float path; contradicts ADR 0010 IFFT evidence at volume scale.
- **If chosen:** Document Energy as **out of scope for cross-language bit-identical floats** in Phase 1; pursue MATLAB MKL integration or accept Phase 1 blocked on Energy only.

### Option B — Split structural / float gates (recommended pending review)

- **Structural gate (strict):** `scale_indices`, shapes, `lumen_radius_microns`, and all non-Energy stages unchanged — strict `prove-exact`.
- **Float gate (bounded ULP):** `energy.energy` passes when, for all voxels with matching `scale_indices`, ULP distance ≤ **`max_ulps`** (proposed default: **48** for crop + canonical; **8** retained for one-voxel probes only).
- **Implementation:** Promote `prove-energy-ulp` to a **certification sub-gate** wired into `prove-exact --stage energy` (or `prove-exact-sequence`) with reported `max_ulps` in `exact_proof_energy.json`; keep `prove-exact` strict mode behind `--strict-floats` for regression.
- **Pros:** Aligns certification with measured cross-library floor; crop v2 already **99.11%** at 48 ULP with **0** scale errors; unblocks downstream parity work.
- **Cons:** Weakens literal R1 wording; requires spec R1 amendment and explicit claim boundary in [PARITY_CERTIFICATION_GUIDE.md](../reference/workflow/PARITY_CERTIFICATION_GUIDE.md).
- **Residual risk:** 37k voxels at 48 ULP are denormal-dominated — may need a secondary rule (e.g. fail if \|Δ\| > 1e-10 at \|energy\| ≥ 1e-3) if those voxels matter topologically.

### Option C — Tiered tolerance (crop loose, canonical strict)

- **Crop harness:** Option B with `max_ulps=48`.
- **Canonical `180709_E`:** Option A strict `np.equal` until a separate canonical investigation completes.
- **Pros:** Unblocks crop iteration without lowering canonical claim bar.
- **Cons:** Crop success may not predict canonical Energy closure; two certification semantics to maintain.

### Option D — MATLAB-produced Python checkpoint (reference runner)

- **Gate:** Python compares against oracle using **promoted MATLAB HDF5 only**; certification requires running Python through a **MATLAB-invoked** or **MKL-matched** float path for Energy, not the current NumPy FFT stack.
- **Pros:** Theoretically achieves strict `np.equal` without relaxing R1.
- **Cons:** High engineering cost; may still hit libm/BLAS differences outside FFT; long schedule risk for Phase 1.

## Recommendation

**Accept Option B** with `max_ulps=48` on crop and canonical Energy `prove-exact`, **strict `scale_indices` unchanged**, and a documented residual denormal policy (absolute delta cap on scale-agreeing mismatches where \|oracle energy\| ≥ 1e-3).

Rationale:

1. Scale winners — the topological driver for downstream Vertices/Edges — are **strict-zero** on v2.
2. Float drift is **bounded** (max \|Δ\| ≈2×10⁻¹¹) and consistent with ADR 0010.
3. Option A leaves Phase 1 blocked with no identified localized Python fix.
4. Option C defers the same canonical question; Option D is a multi-quarter platform bet.

If Option B is rejected, **explicitly Accept Option A** and record Phase 1 Energy certification as **deferred** with ADR 0010 + crop ULP triage as the evidence record.

## Considered Options (summary)

| Option | Energy float gate | Scale gate | Phase 1 crop unblock |
| --- | --- | --- | --- |
| A Status quo | strict `np.equal` | strict | Unlikely |
| B Split gates | ULP ≤ 48 (proposed) | strict | Yes (~99.11% + policy tweak for denormals) |
| C Tiered | strict canonical / ULP crop | strict | Partial (crop only) |
| D MKL/reference runner | strict `np.equal` | strict | Unknown schedule |

## Consequences (when Accepted)

- Update [phase-1-exact-route-spec.md](../plans/phase-1-exact-route-spec.md) R1/R acceptance examples for the chosen option.
- Update [PARITY_CERTIFICATION_GUIDE.md](../reference/workflow/PARITY_CERTIFICATION_GUIDE.md) and [PARITY_PRE_GATE.md](../reference/workflow/PARITY_PRE_GATE.md) with the Energy float rule.
- Wire chosen gate into `ExactProofCoordinator` / `artifact_comparator` or sequence orchestration (Option B).
- Record evidence baselines in [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md) (`energy_ulp_triage_v2.json`, `exact_proof_energy_ulp.json`).
- **Tolerance generalizes to all continuous float fields.** Implemented as a comparator `float_tol` `(rtol=1e-7, atol=1e-9)`: every floating-point field (e.g. `lumen_radius_microns`, vertex/edge `energies`) certifies within `np.allclose`, while integer/topological fields (`scale_indices`, `positions`, `connections`) stay strict. `energy.energy` keeps its dedicated scale-aware gate. `--strict-floats` sets `float_tol=None` to force bit-identical comparison everywhere (regression).
- **Vertex energies are sourced from the raw `vertices.mat`** (true physical energy), since MATLAB curation overwrites `curated_vertices.mat` energies with a rank ramp. Positions + scales still come from the curated (post-choose) artifact.

## Evidence references

- Triage: `workspace/scratch/energy_ulp_triage_v2.json`
- Strict proof: `workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/exact_proof_energy.json`
- Advisory ULP: `workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/exact_proof_energy_ulp.json`
- Oracle: `workspace/oracles/180709_E_crop_M_v2` (`batch_260624-105705`)
- CLI: `slavv parity prove-energy-ulp --max-ulps N` (advisory until promoted)