---
title: "Phase 2: Pythonic Optimization & Scalability Spec"
type: spec
status: ideation
date: 2026-06-24
topic: pythonic-optimization-scalability
---

# Phase 2: Pythonic Optimization & Scalability Spec

**Author**: Engineering Team
**Prerequisite**: Phase 1 exact-route certification under ADR 0011/0012, with the
canonical baseline frozen per
[phase-1-to-phase-2-transition-spec.md](phase-1-to-phase-2-transition-spec.md)
**Research input**: [Post-parity optimization & the translation paper](../research/post-parity-optimization-and-paper.md)
— cited methodology, a bit-preserving vs bit-perturbing technique matrix, and
publication guidance.

---

## 1. Executive Summary

Phase 1 establishes trust by comparing Python SLAVV against preserved MATLAB
oracles with stage-specific certification bars: ADR 0011 for Energy and
Vertices, and evaluated ADR 0012 spatial/topology bars for Edges and Network.
To reach those bars, the Python codebase intentionally preserves MATLAB-facing
behaviors such as Fortran-order tie-breaking, rounding choices, and edge-case
handling where they affect certified outputs.

**Phase 2 represents the transition from emulation to optimization.** With a
frozen Phase 1 canonical baseline and exact-proof harness, Phase 2 can unwind
MATLAB-compatibility constraints behind explicit regression gates. The goal is
to return SLAVV to idiomatic, performant Python (C-order) and leverage modern
data science ecosystems (e.g., SciPy, Numba, PyTorch, and CuPy) to achieve
10x-100x speedups without losing scientific/topological equivalence.

## 2. Core Objectives

1.  **Reclaim Idiomatic Python Performance:** Remove the manual `[Y, X, Z]` grid alignment and Fortran (`order="F"`) contiguity requirements. Return the engine to native `[Z, Y, X]` C-order processing to eliminate cache misses and massive transposition overhead.
2.  **Unshackle the Ecosystem:** Replace bespoke MATLAB shims (e.g., `_interp3_matlab_linear_inf`, `_matlab_round`, `_matlab_zero_based_linspace`) with heavily optimized, C-backed standard library equivalents (`scipy.ndimage.map_coordinates`, `numpy.round`, `numpy.linspace`).
3.  **Hardware Acceleration:** Introduce GPU acceleration (via PyTorch or CuPy) for the most expensive mathematical operations (e.g., multi-scale FFTs, Batched EIGH/Hessian filtering).
4.  **Parallelize the Frontier:** Break the strict deterministic lockstep of the global watershed priority queue to allow for multi-threaded or distributed edge discovery.

## 3. The Path Forward: Planned Initiatives

### 3.1 Memory Layout Unwinding (The "Great Transpose")
*   **Problem:** Phase 1 forces the entire pipeline into Fortran-order `[Y, X, Z]` arrays simply to ensure MATLAB's `find()` tie-breaking logic is matched.
*   **Action:** Refactor all stages to operate natively on C-order `[Z, Y, X]` numpy arrays. 
*   **Validation:** Use the frozen Phase 1 canonical outputs as the baseline.
    Phase 2 outputs are expected to differ in strict emission order and
    tie-breaking details, but must satisfy the Phase 2 topology and spatial
    regression gates.

### 3.2 Standard Library Re-Integration
*   **Action:** Systematically audit and replace `matlab_*.py` parity ports with native equivalents where topological tolerance allows. 
    *   Replace exact mesh generation with native `np.meshgrid` and standard `np.linspace`.
    *   Replace `_matlab_round` with native `np.rint` or standard `round`.
*   **Risk:** Edge cases near volume boundaries may generate slightly different
    candidate sets. Phase 2 needs an explicit topology/spatial regression gate
    before replacing any parity shim on the production path.

### 3.3 GPU-Accelerated Hessian Filtering
*   **Problem:** The Energy stage relies on CPU-bound FFTs and a batched `np.linalg.eigh` loop that takes over an hour for canonical volumes.
*   **Action:** Implement a `torch` or `cupy` backend for `matlab_energy_filter_v200.py`. Move the entire multi-scale filtering loop (FFT -> Derivative Kernels -> Hessian Eigh) to the GPU.
*   **Target:** Reduce Energy stage execution time from 90+ minutes to < 5 minutes.

### 3.4 Distributed Edge Discovery
*   **Problem:** The global watershed algorithm (`matlab_get_edges_by_watershed.py`) is locked to a single thread because it relies on a shared, global `FrontierQueue` and deterministic tie-breaking.
*   **Action:** Investigate chunk-based local watershed discovery with a map-reduce style border-stitching phase, or port the frontier expansion to a compiled Numba/Cython kernel that releases the GIL.

## 4. Phase 2 Gating Criteria
Unlike Phase 1, Phase 2 cannot use strict zero missing/extra comparisons against the MATLAB oracle.

**Proposed Metrics for Phase 2 Acceptance:**
1.  **Topological Isomorphism:** The resulting network graph (Nodes + Edges) must have identical connectivity and topology to the Phase 1 canonical graph, allowing for < 1% spatial drift (in microns) for bridge vertices.
2.  **Performance:** >10x reduction in end-to-end execution time for the 512x512x64 canonical volume.
3.  **Memory:** Peak memory footprint must remain < 16GB, but preferably < 8GB to allow parallel processing of multiple chunks.
