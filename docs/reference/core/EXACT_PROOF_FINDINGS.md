# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-05-28

This document serves as the **Authoritative Status Log** for aligning Python's native vectorization output perfectly with the original MATLAB mathematics (the "Exact Proof").

---

## 📊 Executive Status Summary

The goal is 100% mathematical parity against the canonical MATLAB oracle. 

| Workflow Stage | Parity Status | Immediate Blocker |
| :--- | :--- | :--- |
| **NATIVE ENERGY** | ✅ **COMPLETED** | None (Canonical exact-compatible) |
| **VERTICES** | ✅ **VERIFIED** | Successfully certified downstream |
| **EDGES** | 🟡 **IN PROGRESS** | Phase 1 cert run + parity closure (strict zero pairs) |
| **NETWORK** | ⏳ **PENDING** | Phase 1 `init-exact-run --stop-after network` |
| **prove-exact energy** | ✅ **HARNESS** | `EXACT_STAGE_ORDER` includes `energy` (2026-05-28) |

---

## 🏆 High-Water Mark Breakthrough (May 2026)

A major architectural breakthrough was achieved in May 2026, dramatically narrowing the discrepancy gap in edge candidate generation.

**Phase 1 certification run (active)**: `workspace/runs/oracle_180709_E/phase1_cert_network` — native `init-exact-run --stop-after network`, `npy` energy storage, oracle `180709_E_batch_190910-103039`. Log: `workspace/runs/phase1_cert_network.log`.

**Champion experiment path (edges baseline)**: `workspace/runs/oracle_180709_E/validation_strel_fix_output_v29`

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

## 🚀 Active Blockers & Immediate Next Steps

While the 88.7% match rate is a milestone, we must close the remaining **135 missing pairs**.

1.  **Certification v2.0**: Complete the ongoing certification run on `180709_E` to verify bit-accurate tie-breaking.
2.  **Hub Vertex Complexities**: Audit high-degree junction exploration logic where branching decisions diverge near high-density clusters.
3.  **Run Final Proof Loop**: Once Edge closures improve beyond 95%, execute `prove-exact --stage all` to lock down full pipeline certification.

