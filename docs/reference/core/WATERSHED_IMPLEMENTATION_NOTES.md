# Global Watershed Implementation Notes

[Up: Reference Docs](../README.md)

This document provides technical implementation details for the global watershed algorithm in SLAVV Python, with a focus on MATLAB parity and internal architecture.

---

## 🏗️ Architecture Overview

The watershed discovery is implemented as a single-pass discovery over shared spatial maps. The implementation is located in `slavv_python/pipeline/edges/matlab_get_edges_by_watershed.py` (MATLAB port of `get_edges_by_watershed.m`). Heap/claim helpers live in `matlab_watershed_heap.py`; strel LUT geometry in `matlab_calculate_linear_strel_range.py`.

### Modular Decomposition
To maintain readability, the 800+ line discovery logic is decomposed into specialized helpers:
1.  `_initialize_matlab_global_watershed_state`: Builds the initial spatial maps (vertex indices, borders, energies).
2.  `_matlab_global_watershed_prepare_size_map`: Orchestrates the scale-aware `size_map` from input labels and vertex scales.
3.  `_matlab_global_watershed_current_strel`: Extracts a local neighborhood (strel) and its metadata (offsets, pointer indices, distances).
4.  `_matlab_global_watershed_reveal_unclaimed_strel`: Claims voxels in the spatial maps for a vertex.
5.  `_matlab_global_watershed_insert_available_location`: Manages the frontier priority queue (worst-to-best sorted list).
6.  `_matlab_global_watershed_reset_join_locations`: Handles frontier cleanup during watershed joins.
7.  `_matlab_global_watershed_assemble_results`: Finalizes traces and assembles the candidate payload.

---

## 🧠 Shared State Management

The algorithm relies on five primary 3D spatial maps, all using **Fortran-order (F)** layout to match MATLAB linear indexing.

### Internal Grid Alignment
To achieve bit-perfect MATLAB parity, the engine realigns the physical volume (Standard: [Z, Y, X]) to an internal **[Y, X, Z]** grid. 

- **Rationale**: MATLAB's column-major memory layout prioritizes the first dimension (Y). By aligning the Python internal grid so that Y is the first index and using Fortran contiguity, we ensure that:
    1.  **Linear Indexing**: `ravel(order="F")` linear indices physically match MATLAB's `find()` results.
    2.  **Tie-Breaking**: Bit-accurate `argmin` on linear indices correctly prioritizes the Y-axis, then X, then Z, mirroring MATLAB's exact deterministic discovery order.
    3.  **Rounding Consistency**: FFT and interpolation traversal match MATLAB's loop nesting, minimizing floating-point drift.

| Map | Type | Description |
| :--- | :--- | :--- |
| `vertex_index_map` | `uint32` | Stores 1-based vertex IDs (0 = unclaimed). |
| `pointer_map` | `uint64` | Stores 1-based indices into the scale-specific strel LUT. |
| `energy_map_temp` | `float64`[^dtype] | Stores the *original* (unpenalized) energies for frontier sorting. |
| `d_over_r_map` | `float64` | Accumulates normalized distances ($r/R$) along traces. |
| `branch_order_map` | `uint8` | Tracks the branch depth from the origin vertex. |

[^dtype]: Core watershed maps (including `energy_map_temp` in `VoxelClaimMap`, see `slavv_python/pipeline/edges/matlab_watershed_heap.py`) are computed in `float64` per the double-precision alignment fix to avoid tie-breaking divergence; persisted energy volumes remain `float32`. See [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) and [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md).

---

## ⚖️ Parity Details

### Common Divergence Patterns

#### 1. Distance Normalization (r/R)
MATLAB often uses relative distances normalized by the vessel radius ($R$) at each step.
-   **Divergence**: Python used absolute micron distances for energy penalties.
-   **Fix**: Use `r/R` ratios for local distance and size tolerances.

#### 2. Energy Map Integrity
The global watershed algorithm maintains shared state.
-   **Divergence**: Python incorrectly wrote penalized (suppressed) energies back to the shared map.
-   **Truth**: MATLAB uses unpenalized original energies for frontier sorting (`energy_map_temp`). Penalties are applied locally during seed selection only.

#### 3. Iterative Directional Suppression
-   **Divergence**: A previous finding incorrectly suggested suppression was outside the seed loop.
-   **Truth**: Suppression **is** iterative inside the `seed_idx` loop. Each chosen seed suppresses the local field for subsequent seeds of the same vertex.

### Two-Tier Penalty System
MATLAB applies penalties in two distinct stages:
1.  **Discovery (Static)**: Size, absolute distance, and initial direction penalties are applied *once* per strel expansion to find the best primary seed.
2.  **Seed Loop (Iterative)**: Directional suppression is applied *within* the seed loop. Each chosen seed suppresses the local energies for subsequent seeds of the same vertex.

### F-Contiguity and Performance
To ensure writes to `ravel(order="F")` views persist in the underlying 3D arrays, all maps **must** remain F-contiguous.
-   **Risk**: Operations like `np.clip` or `np.asarray(copy=True)` may return C-contiguous arrays.
-   **Mitigation**: The implementation uses `np.asfortranarray()` or explicit `order="F"` allocations to maintain contiguity.

---

## 🧪 Verification

Exact parity is verified using 3x3x3 and 5x5x5 synthetic volumes in `tests/unit/pipeline/test_global_watershed_comprehensive.py`. These tests verify:
-   Exact pointer values match MATLAB LUT indexing.
-   Frontier insertion order matches MATLAB's descending energy priority.
-   Join logic correctly removes specific indices from the frontier.

## Debugging Exact-Route Gaps

When `edges.connections` remains red after upstream gates pass, start from the
released MATLAB source and the maintained proof artifacts:

- Measure with `slavv parity prove-exact` or
  `prove-exact-sequence`; do not use informal match rates as an acceptance bar.
- Audit against `external/Vectorization-Public/source/get_edges_by_watershed.m`.
  Commented MATLAB code is not behavior; verify the line is active before porting
  an idea.
- For a specific missing pair, compare `current_strel_energies`, adjusted local
  energies, chosen `strel_idx`, and `available_locations` insertion/removal.
- Preserve MATLAB indexing assumptions: 1-based values at the MATLAB boundary,
  Fortran-order linear priority, and `ravel(order="F")` for Python comparisons.
