# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-05-12

This document serves as the **Authoritative Status Log** for aligning Python's native vectorization output perfectly with the original MATLAB mathematics (the "Exact Proof").

---

## 📊 Executive Status Summary

The goal is 100% mathematical parity against the canonical MATLAB oracle. 

| Workflow Stage | Parity Status | Immediate Blocker |
| :--- | :--- | :--- |
| **NATIVE ENERGY** | ✅ **COMPLETED** | None (Canonical exact-compatible) |
| **VERTICES** | ✅ **VERIFIED** | Successfully certified downstream |
| **EDGES** | 🚧 **IN PROGRESS** | Candidate-generation & Frontier ordering |
| **NETWORK** | ⏳ **PENDING** | Awaiting upstream Edge closure |

---

## 🏆 High-Water Mark Breakthrough (May 2026)

A major architectural breakthrough was achieved in May 2026, dramatically narrowing the discrepancy gap in edge candidate generation.

**Champion Experiment Path**: `workspace\runs\oracle_180709_E\validation_strel_fix_output_v8`

### The Solution: Universe Realignment & Vertex Coverage
- **Issue 1**: Axis transposition in `generate.py` was `(0, 2, 1)` (swapping X and Y) instead of the required `(2, 1, 0)` to reach the `[Z, X, Y]` orientation expected by the engine. This caused spatial misalignment and out-of-bounds sampling.
- **Issue 2**: The `curated_vertices` artifact in the oracle was an incomplete subset (1313 nodes), causing many valid oracle connections to refer to missing vertices in Python.
- **Fix**: Corrected axis transpose to `(2, 1, 0)` and implemented fallback to the full vertex set (1380 nodes) embedded in the oracle's `edges` artifact.
- **Outcome**: Successfully resolved Hub Vertex 1350 and boosted the match rate to **77.7%**, nearly reaching the 80% milestone.

### Final Mathematical Impact
| Metric | Previous Baseline | Current High-Water Mark | Improvement |
| :--- | :--- | :--- | :--- |
| **Matched MATLAB Pairs** | 670 | **930** | **+39% Increase** |
| **Total Match Rate** | 56.0% | **77.7%** | **Near Milestone** |
| Missing Pairs | 527 | 267 | 🔻 Reduced by 260 |
| Over-generated Pairs | 355 | 258 | 🔻 Reduced by 97 |

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
- `edge_number_tolerance = 2`

---

## 🛠️ Verified Infrastructure Fixes

The core codebase has absorbed the following permanent fixes, ensuring structural stability:

*   ✅ **Backtracking Pointer Correction**: Fixed reverse-index logic, allowing trace recovery back to the origin vertex.
*   ✅ **Stable Discovery Sorting**: Forces deterministic processing orders matching MATLAB's explicit energy quality sorting.
*   ✅ **Trace Order Randomization**: Anchored native shuffling to a stable, reproducible seeded RNG generator.
*   ✅ **Distance Normalization (r/R)**: Corrected physical energy penalties to scale relatively to the vessel's radius ($R$).
*   ✅ **Strel Offset Alignment**: Realigned watershed structuring element (strel) offsets and loops to match the (Z, X, Y) universe layout, fixing major distance-penalty errors.
*   ✅ **Filtering Logic Reordering**: Realigned the cleanup sequence (Crop first $\rightarrow$ Pair Cleanup second) to protect valid pairs from phantom blocking.

---

## 🚀 Active Blockers & Immediate Next Steps

While the 56% match rate is a milestone, we must close the remaining **527 missing pairs**.

1.  **Frontier Ordering Divergence** (High Priority): Investigate fine-grained seed selection priority mismatches within the edge watershed crawler.
2.  **Hub Vertex Complexities**: Audit high-degree junction exploration logic where branching decisions diverge near high-density clusters.
3.  **Run Final Proof Loop**: Once Edge closures improve beyond 95%, execute `prove-exact --stage all` to lock down full pipeline certification.
