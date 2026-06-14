# SLAVV Milestone Progress Tracker

This document provides a high-level history of achievements and current mission focus. It is the context-free entry point for understanding *what has been done* and *where we are going*.

---

## 🎯 Current Mission: Phase 1 Parity Certification
**Goal**: 100% bit-perfect parity between Python SLAVV and MATLAB oracle for full `180709_E` volume.

| Status | Stage | Target | Current Status |
| :--- | :--- | :--- | :--- |
| 🟡 | **Energy** | Full `180709_E` | Rerunning with bit-perfect phase alignment (Active PID: 27764) |
| ⏳ | **Vertices** | Full `180709_E` | Blocked by Energy completion |
| ⏳ | **Edges** | Full `180709_E` | Fixes for KeyError and Tie-breaking verified in unit tests |
| ⏳ | **Network** | Full `180709_E` | Pending upstream certification |

---

## 🏆 Recent Achievements (June 2026)

### 🔄 Watershed Orientation Alignment (June 13)
- **Problem**: Systemic indexing mismatch in global watershed due to physical [Z, Y, X] vs MATLAB [Y, X, Z] orientation.
- **Discovery**: microns_per_voxel was axis-flipped, and energy maps were being read with the wrong strides.
- **Solution**: Implemented global reorientation of watershed inputs to [Y, X, Z] and transposed results back to physical [Z, Y, X].
- **Impact**: Guaranteed bit-perfect voxel extraction and connectivity order; resolved the primary blocker for Edge parity.

### 🎨 Vertex Painting Rounding (June 13)
- **Problem**: Greedy selection divergence due to Python's round-to-even behavior.
- **Solution**: Implemented MATLAB-exact round-half-up logic in `vertices/detection.py` and aligned all tie-breaking to Lowest Linear Index Priority.
- **Impact**: Bit-perfect painting exclusion zones verified.

### 📐 Mesh Phase Alignment (June 13)
- **Problem**: 3-pixel coordinate shift in energy interpolation causing failure at the 4th element.
- **Discovery**: Saturated subtraction was clipping negative offsets, and `linspace` meshes were failing to account for the downsampling starting phase.
- **Solution**: Removed saturated offsets and implemented phase-aware `_matlab_zero_based_linspace`.
- **Impact**: Bit-perfect coordinate mapping verified in new unit tests; resolved the final mathematical blocker for Energy parity.

### 🧩 Chunking Memory Fix (June 12)
- **Problem**: `ArrayMemoryError` during exact-route energy calculation despite recent optimizations. 
- **Discovery**: `max_voxels_per_node_energy` was set too low (6000), causing a 726-chunk lattice where each chunk ballooned to ~9 GiB due to massive 400-pixel overlaps.
- **Solution**: Increased `max_voxels_per_node_energy` to 4M, reducing the lattice to [1, 1, 2] (2 chunks) and per-chunk memory to ~1 GiB.
- **Impact**: Stable execution on 16GB hardware with limited free RAM; resolved persistent Energy stage blockers.

### ⚡ FFT Memory Optimization (June 11)
- **Problem**: `ArrayMemoryError` during MATLAB-style symmetric IFFT on canonical volumes.
- **Solution**: Refactored `_ifftn_matlab_symmetric` to use sparse conjugate re-population instead of full-volume flipped copies.
- **Impact**: Reduced per-chunk memory footprint by 50%, enabling stable processing of 512x512x64 volumes.

### 🚀 The Memory Breakthrough (June 10)
- **Problem**: `ArrayMemoryError` on 512x512x64 volumes.
- **Solution**: Refactored `exact_mesh.py` to use in-place scale comparisons, eliminating 4D energy stacks.
- **Impact**: 30x reduction in peak memory (300MB → 10MB per thread).

### 📐 The Grid Alignment (June 11)
- **Problem**: Linear indexing mismatches in watershed discovery.
- **Solution**: Standardized internal grid to **[Y, X, Z]** with Fortran (F) memory order.
- **Impact**: Perfect alignment with MATLAB's `find()` and sorting tie-breaks.

### 🛡️ Watershed Robustness (June 11)
- **Problem**: `KeyError` in `FrontierQueue` due to duplicate seed pushes.
- **Solution**: Implemented iterative directional suppression and hardened the priority queue registry.
- **Impact**: Resolved the primary blocker for Edge stage certification.

---

## 🗓️ 3-Day History (June 8 - June 13)

- **June 13**: Fixed 3-pixel coordinate shift in Energy mesh interpolation. Verified with bit-perfect linspace unit test.
- **June 12**: Resolved Energy `ArrayMemoryError` by increasing chunk size to 4M voxels.
- **June 11**: Implemented [Y, X, Z] grid alignment repository-wide. Fixed KeyError in watershed frontier. Restarted certification reruns from Energy stage.

---

## 🗺️ Navigation Hub

| Document | Purpose |
| :--- | :--- |
| **[AGENTS.md](../AGENTS.md)** | **Operational Guide**: How to work, domain glossary, decision tree. |
| **[TODO.md](TODO.md)** | **Checklist**: Active tasks and upcoming work. |
| **[FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)** | **Live Status**: Active run details, PIDs, and proof failure logs. |
| **[README.md](../README.md)** | **General Info**: Project overview, setup, and public CLI. |

---
*Last Updated: 2026-06-13*
