# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-05-28

**Authoritative status log** for exact-parity alignment with MATLAB. **Live operational status** (active runs, proof failures, blockers) lives here—not in [TODO.md](../../TODO.md). Tasks and checkboxes: TODO only.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---

## 📊 Executive status (stage model)

Phase 1 exit criterion: **strict zero** missing/extra per stage via sequential `prove-exact` (energy → vertices → edges → network) on full `180709_E`—not informal match-rate thresholds.

| Stage | Harness / prior work | Phase 1 certification (strict zero) |
| :--- | :--- | :--- |
| **Energy** | Native Hessian path exact-compatible | 🟡 **Blocked** on crop harness (value mismatch); canonical run in progress |
| **Vertices** | Verified on prior surfaces | ⏳ Pending passing proof on certified run |
| **Edges** | v29 ~88.7% pair match (diagnostic baseline) | ⏳ Pending sequential proof after upstream stages |
| **Network** | End-to-end pipeline runs | ⏳ Pending sequential proof |
| **`prove-exact` energy stage** | ✅ In `EXACT_STAGE_ORDER` (2026-05-28) | Required for R3 gating |

---

## 🚦 Active Phase 1 operations

| Track | Run / artifact | Status |
|-------|----------------|--------|
| **Canonical cert** | `workspace/runs/oracle_180709_E/phase1_cert_network` | Pipeline in progress or resume — **one writer** on this dest root. Oracle: `180709_E_batch_190910-103039`. Monitor: `python scripts/cli/monitor_run_progress.py --run-dir workspace/runs/oracle_180709_E/phase1_cert_network` |
| **Crop harness oracle** | `workspace/oracles/180709_E_crop_M` | ✅ Promoted ([HDF5 energy loader](../../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md)) |
| **Crop harness run** | `workspace/runs/oracle_180709_E/crop_M_exact` | Pipeline ✅ complete. **`prove-exact-sequence` failed on energy:** same shape `[64,256,256]`, sample values MATLAB ≈ −20 vs Python ≈ −0.004 |

**Champion edges baseline (informal, not cert bar):** `workspace/runs/oracle_180709_E/validation_strel_fix_output_v29`

### Operator commands

```powershell
# Monitor canonical run
python scripts/cli/monitor_run_progress.py --run-dir workspace/runs/oracle_180709_E/phase1_cert_network

# Resume canonical (single process)
python scripts/cli/parity_experiment.py resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --stop-after network --skip-preflight

# Re-prove crop harness after energy fix
python scripts/cli/parity_experiment.py prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M
```

---

## 📚 Compound learnings (parity-related)

Curated index of solved problems under `docs/solutions/` (from `/ce-compound`). Search all solutions via YAML frontmatter (`module`, `tags`, `problem_type`); see [docs/solutions/README.md](../../../solutions/README.md).

| Topic | Doc |
|-------|-----|
| MATLAB energy HDF5 + `promote-oracle` | [matlab-v200-energy-hdf5-oracle-loader.md](../../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md) |

_Add rows here when a new compound doc is parity-relevant; do not duplicate full write-ups in this file._

---

## 🏆 High-Water Mark Breakthrough (May 2026)

A major architectural breakthrough was achieved in May 2026, dramatically narrowing the discrepancy gap in edge candidate generation.

_See [Active Phase 1 operations](#-active-phase-1-operations) for current run paths and status._

### The Solution: Parameter Alignment & NaN Stability
- **Parameter Alignment (v29)**: Discovered that the MATLAB oracle was generated with `edge_number_tolerance = 4`, while Python was hardcoded to 2. Aligning this parameter allowed high-degree vertices (hubs) to initiate sufficient exploratory traces.
- **NaN Stability**: Fixed a floating-point instability where multiplying `-Inf` (vertex priority) by `0.0` (directional suppression factor) created `NaNs`, leading to incorrect seed selection in subsequent iterations.
- **Precision Alignment (May 22)**: Implemented bit-accurate tie-breaking using exact equality (`==`) and Fortran-order linear index priority. Removed all remaining `float32` casts in the expansion frontier.
- **Tightened Filtering**: Implemented hard distance cutoffs ($d/R > 3.0$) and aligned edge influence sigmas to exactly $2/3$.
- **Outcome**: Successfully reached the **88.7%** match rate milestone (1062/1197 pairs). Certification run v2 is underway to verify the impact of bit-accurate tie-breaking.

### Final Mathematical Impact
| Metric | Previous Baseline (v28) | High-Water Mark (v29) | Current (v2.0) |
| :--- | :--- | :--- | :--- |
| **Matched MATLAB Pairs** | 958 | 1062 | *Pending Run* |
| **Total Match Rate** | 80.0% | 88.7% | *Pending Run* |
| Missing Pairs | 239 | 135 | *Pending Run* |
| Over-generated Pairs | 263 | 371 | *Pending Run* |

---

## ⚖️ Exact Parameter Fairness Gate

To guarantee a fair mathematical race, all exact-route experiments must maintain structural lock-step between Python and MATLAB configuration inputs. This is validated via the **Parameter Diffusion Matrix**.

Every compliant proof run maintains three persistent JSON manifests under `01_Params/`:

1. **`shared_params.json`**: The authoritative overlap of settings that must exist in both MATLAB and Python.
2. **`python_derived_params.json`**: Internal Python-only pipeline management levers.
3. **`param_diff.json`**: The cryptographic hash bridge that proves zero illicit divergence occurred between the split configuration states.

### Locked Mathematical Constants
The audit system mandates these exact value bindings (derived from source-hardcoded MATLAB constants):
- `step_size_per_origin_radius = 1.0`
- `max_edge_energy = 0.0`
- `distance_tolerance_per_origin_radius = 3.0`
- `edge_number_tolerance = 4` (Corrected from 2)

---

## 🛠️ Verified Infrastructure Fixes

The core codebase has absorbed the following permanent fixes, ensuring structural stability:

*   ✅ **Double-Precision Energy Alignment**: Forced all core watershed maps (`energy_map_temp`, `vertex_energies`) and neighborhood penalty calculations to `float64`. This prevents precision-induced tie-breaking divergences where `float32` would collapse distinct energy values into identical bits, causing different seed selections than MATLAB's `double`.
*   ✅ **Bit-Accurate Tie-Breaking**: Replaced `np.isclose` with exact bitwise equality and added linear index priority to the frontier priority queue, matching MATLAB's hub vertex exploration behavior.
*   ✅ **Hard Distance Cutoff**: Implemented the MATLAB-exact $d/R > 3.0$ expansion cutoff in the watershed loop.
*   ✅ **Edge Influence Alignment**: Updated default `sigma_per_influence_edges` to $2/3$, aligning with MATLAB's conflict painting regions.
*   ✅ **Stable Frontier Splicing**: Verified and anchored the `available_locations` insertion logic to mirror MATLAB's `find(..., 'last')` and `find(..., 'first')` behavior, ensuring stable FIFO/LIFO handling for identical energy seeds.
*   ✅ **Backtracking Pointer Correction**: Fixed reverse-index logic, allowing trace recovery back to the origin vertex.
*   ✅ **Stable Discovery Sorting**: Forces deterministic processing orders matching MATLAB's explicit energy quality sorting.
*   ✅ **Trace Order Randomization**: Anchored native shuffling to a stable, reproducible seeded RNG generator.
*   ✅ **Distance Normalization (r/R)**: Corrected physical energy penalties to scale relatively to the vessel's radius ($R$).
*   ✅ **Strel Offset Alignment**: Realigned watershed structuring element (strel) offsets and loops to match the (Z, X, Y) universe layout, fixing major distance-penalty errors.
*   ✅ **Filtering Logic Reordering**: Realigned the cleanup sequence (Crop first $\rightarrow$ Pair Cleanup second) to protect valid pairs from phantom blocking.

---

## 🚀 Active blockers

1. **Crop harness energy proof** — `prove-exact-sequence` fails at energy on `crop_M_exact` (magnitude mismatch). Blocks tier-2 pre-gate until resolved (params, sign/scale, or HDF5 plane semantics).
2. **Canonical run completion** — Finish `phase1_cert_network` through network with a single writer; then run `prove-exact-sequence` on full `180709_E`.
3. **Sequential strict-zero closure** — After energy passes on a run, prove vertices → edges → network in order. v29 **135 missing / 371 extra** pairs remain the informal edges baseline until `prove-exact --stage edges` reports zero.

**Superseded guidance:** “>95% match” or “prove-exact once parity exceeds 95%” is not the Phase 1 bar. Use strict zero per stage only.

