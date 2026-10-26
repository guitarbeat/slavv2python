# Phase 2 Performance Profiling & Bottleneck Analysis Report

**Document Type:** Technical Architecture & Performance Report  
**Workstream:** Phase 2 Optimization & Scalability  
**Authors:** SLAVV Performance & Engineering Team  
**Date:** October 26, 2026  
**Status:** Validated on Disk  
**Scope:** Canonical Full Volume ($512 \times 512 \times 64$, 16.8M voxels) & Crop Harness ($256 \times 256 \times 64$)

---

## 1. Executive Summary

Phase 1 established mathematical trust by achieving certified 1:1 structural exact parity against legacy MATLAB oracles (ADR 0011 / ADR 0012). With the Phase 1 canonical baseline frozen (`canonical_full_v18`), **Phase 2 marks the transition from emulation to optimization**.

This report provides the first comprehensive, end-to-end performance profile of the entire SLAVV pipeline across all four stages, disaggregating sub-stage spans and uncovering the exact micro-architectural bottlenecks within the primary runtime driver: **Watershed Edge Discovery**.

```mermaid
pie title Phase 2 Stage Time Distribution (Full Volume Canonical R2024b)
    "Energy (Parallel Chunks, 6 Workers)" : 9342
    "Edges (Watershed Discovery + Selection)" : 764
    "Vertices (Maxima & Structuring Element Painting)" : 108
    "Network (CSR Adjacency & Strand Decomposition)" : 23
```

---

## 2. Macro Pipeline Benchmarks: Cross-Platform Performance

Prior to this work, the Phase 2 profiling baseline (`docs/reference/core/phase2-profiling-baseline.json`) only recorded timings from `canonical_full_v18` on Windows, where Energy and Vertices were carried lineage ($0.0\text{ s}$). 

With the verified completion of `canonical_full_r2024b_v1`, we now have **complete, measured wall-clock metrics across every stage on identical hardware**:

| Pipeline Stage | Windows Baseline (`v18`) | Apple Silicon macOS (`r2024b_v1`) | Speedup Factor | Peak RAM (macOS) | Dominant Sub-Operation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01. Energy** | Unmeasured (lineage carried) | **9,342.1 s** ($155.7\text{ min}$) | — ($17.1\times$ vs serial) | $2.98\text{ GB}$ | 3D FFTs & batched $3\times 3$ Hessian `eigh` |
| **02. Vertices** | Unmeasured (lineage carried) | **107.7 s** ($1.8\text{ min}$) | — | $2.81\text{ GB}$ | Structuring element coordinate painting |
| **03. Edges** | **5,534.0 s** ($92.2\text{ min}$) | **764.0 s** ($12.7\text{ min}$) | **$7.24\times$ Faster** | $2.45\text{ GB}$ | Watershed flood-fill & frontier queue |
| **04. Network** | **416.0 s** ($6.9\text{ min}$) | **23.2 s** ($0.4\text{ min}$) | **$17.93\times$ Faster** | $1.91\text{ GB}$ | Sparse CSR strand decomposition |
| **Total Pipeline** | **$\mathbf{>50\text{ hours}}$** (est. serial) | **$\mathbf{10,237.0\text{ s}}$** ($\approx \mathbf{2.84\text{ hours}}$) | **$\approx \mathbf{18\times}$ Faster** | $\mathbf{<3.0\text{ GB}}$ | End-to-end full volume execution |

---

## 3. Sub-Stage Disaggregation

### 3.1 Edges Stage Breakdown (Discovery vs Selection)
Using durable timing instrumentation ([`slavv_python/analytics/performance/edge_timing.py`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/slavv_python/analytics/performance/edge_timing.py)), we separated Watershed Candidate Discovery from Candidate Selection & Finalization:

```mermaid
graph LR
    subgraph EdgesStage["Edges Stage: 764.0s Total"]
        direction LR
        Init["Initialization<br/>0.001s (0.0%)"] --> Disc["Watershed Discovery<br/>748.77s (98.01%)"]
        Disc --> Sel["Edge Selection & Finalize<br/>15.21s (1.99%)"]
    end
```

- **Watershed Discovery:** **$748.77\text{ s}$ ($98.01\%$)** — The absolute bottleneck of the Edges stage.
- **Edge Selection & Finalization:** **$15.21\text{ s}$ ($1.99\%$)** — Highly efficient vectorized conflict resolution and Claimed Trace Energy bake ([ADR 0013](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/docs/adr/0013-claimed-energy-trace-provenance.md)).
- **Initialization:** **$0.001\text{ s}$ ($<0.01\%$)**.

### 3.2 Network Stage Breakdown
Extracted from [`canonical_full_r2024b_v1/05_Network/phase2_network_split.json`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/workspace/runs/oracle_180709_E/canonical_full_r2024b_v1/05_Network/phase2_network_split.json):

```mermaid
graph LR
    subgraph NetworkStage["Network Stage: 23.19s Total"]
        direction LR
        Adj["CSR Adjacency<br/>5.43s (23.4%)"] --> Strand["Strand Extraction<br/>10.93s (47.2%)"]
        Strand --> Fin["Finalize & Metrics<br/>6.77s (29.2%)"]
        Fin --> Prune["Hair/Cycle Pruning<br/>0.0001s (0.0%)"]
    end
```

- **Strand Trace Parameterization:** **$10.93\text{ s}$ ($47.2\%$)** — Continuous arc-length sampling and smoothing.
- **Finalize & Graph Metadata:** **$6.77\text{ s}$ ($29.2\%$)** — Geometry calculations and bifurcation indexing.
- **CSR Graph Construction:** **$5.43\text{ s}$ ($23.4\%$)** — `scipy.sparse.csr_matrix` degree mapping.
- **Pruning:** **$<0.0001\text{ s}$ ($<0.01\%$)**.

---

## 4. Deep Micro-Profiling: The Watershed Discovery Hot Spots

To understand why Watershed Discovery consumes 98% of the Edges stage, we executed a full `cProfile` analysis on the crop harness dataset (`crop_M_r2024b`, 13,706 vertices, 19,225 candidates). 

Full profile stats are logged in [`workspace/reports/phase2_profiling/edges_detailed_profile.json`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/workspace/reports/phase2_profiling/edges_detailed_profile.json).

### 4.1 Top Function Spans in Watershed Discovery

| Function | File / Module | Call Count | Total Time (s) | Cumulative Time (s) | % of Discovery Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `_matlab_global_watershed_insert_available_location` | `matlab_watershed_heap.py` | 73,783 | **30.42 s** | **30.52 s** | **39.06%** |
| `remove_first_occurrence` / `reset_join_locations` | `matlab_watershed_heap.py` | 40,382 | **16.93 s** | **23.65 s** | **30.26%** |
| `build_matlab_global_watershed_lut` | `matlab_calculate_linear_strel_range.py` | 291,956 | 0.56 s | **12.24 s** | **15.66%** |
| `build_matlab_local_strel_geometry` | `matlab_calculate_linear_strel_range.py` | 99 | **8.53 s** | **11.30 s** | **14.46%** |
| `<method 'append' of 'list'>` | Built-in | 158,545,130 | **7.03 s** | **7.03 s** | **8.99%** |
| `_matlab_frontier_adjusted_neighbor_energies` | `matlab_get_edges_v300_geometry.py` | 100,467 | 0.04 s | **2.66 s** | **3.40%** |
| `_claim_unowned_strel_arrays_numba_impl` | `matlab_watershed_heap.py` | 40,382 | **0.61 s** | **0.61 s** | **0.78%** |

### 4.2 Key Discovery Findings

```mermaid
pie title Watershed Discovery Time Breakdown (Crop Harness)
    "Queue Insertion (_insert_available_location)" : 39.06
    "Join Reset (remove_first_occurrence)" : 30.26
    "Strel LUT Generation (build_local_strel)" : 15.66
    "List Allocations & Appends" : 8.99
    "Numba Geometric Calculations" : 3.40
    "Numba Voxel Claim Updates" : 0.78
    "Remaining Overhead" : 1.85
```

1. **Queue Maintenance Dominates ($69.32\%$ of Total Discovery Time):**
   - In `SortedFrontier`, maintaining `available_locations` as a sorted Python list via `_matlab_global_watershed_insert_available_location` costs **$39.06\%$** of the time. Every new neighbor incurs a reverse linear search in Python followed by an $O(N)$ `list.insert()` array shift.
   - When active watershed fronts collide, `_matlab_global_watershed_reset_join_locations` removes elements via a full linear scan over tens of thousands of list items, consuming another **$30.26\%$** of runtime.
2. **Numba JIT Successfully Neutralized Geometric & Voxel Costs:**
   - Innovation 3.4 (Numba geometric penalties and voxel claiming) reduced directional suppression and multi-array voxel updates to just **$3.4\%$** and **$0.78\%$** of total time, respectively.
   - This proves that mathematical computation is no longer the bottleneck; the bottleneck has shifted entirely to **queue data structures and memory manipulation**.

---

## 5. Phase 2 Optimization Roadmap

Based on this profiling evidence, Phase 2 optimizations should target the following high-leverage opportunities:

### 1. High-Priority: $O(\log N)$ Bisect or B-Tree Frontier Queue
- **Bottleneck Addressed:** 69.3% queue insertion and join-reset overhead.
- **Strategy:**
  1. Replace the linear reverse-scan in `_matlab_global_watershed_insert_available_location` with standard library `bisect.bisect_left` / `bisect_right` ($O(\log N)$ search).
  2. Implement a `SortedSet` or unrolled B-tree buffer to avoid $O(N)$ Python list shifts during insertion.
  3. Re-evaluate Meyer's Bucket Priority Queue (reverse-engineered from `libmwimagescore.dylib` in Section 3.1 of the translation paper) to achieve $O(1)$ amortized push/pop.
- **Estimated Speedup:** **$3\times - 5\times$ acceleration of Watershed Discovery** (dropping full-volume Edges time from 12.7 min to $<3\text{ min}$).

### 2. High-Priority: Strel Geometry Memoization
- **Bottleneck Addressed:** 15.7% strel lookup and geometry rebuild overhead.
- **Strategy:** Fully pre-compute and store all 99 scale structuring element coordinate offsets and pointer tables in contiguous NumPy arrays prior to watershed loop entry, avoiding repeated function call overhead.

### 3. Medium-Priority: The "Great Transpose" (C-Order Memory Unwinding)
- **Bottleneck Addressed:** Cache misses and array transposition between $[Z, Y, X]$ and $[Y, X, Z]$.
- **Strategy:** Once an explicit Phase 2 regression gate is established, refactor the entire pipeline to native C-contiguous arrays.

### 4. Long-Term: GPU Acceleration for Energy Stage
- **Bottleneck Addressed:** Energy stage is currently 91.3% of total pipeline wall-clock time ($155.7\text{ min}$).
- **Strategy:** Implement PyTorch / CuPy 3D FFTs and batched eigensolvers (`torch.linalg.eigh`), reducing Energy time to $<5\text{ minutes}$.

---

## 6. Post-Optimization Benchmarks

### 6.1 Edges Stage: $10.45\times$ Speedup Validated on Full Volume (ADR 0014)

Following the profiling analysis, we implemented two high-priority optimizations under [ADR 0014](../adr/0014-phase2-optimization-regression-gates.md):
1. **$O(\log N)$ Bisect Frontier Queue & $O(1)$ Presence-Checked Join Resets:** In `SortedFrontier` (`slavv_python/pipeline/edges/watershed/matlab_watershed_heap.py`), replaced linear scans with binary search insertion preserving asymmetric MATLAB tie-breaking, and replaced 500M-element slice copies with $O(1)$ set checks.
2. **Structuring Element Precomputation & Fast Caching:** Precomputed all scale structuring element geometry tables (`watershed_luts` and `watershed_unit_vectors`) prior to watershed loop entry, and vectorized offset generation via `np.ogrid`.

#### Benchmark Results on Canonical Full Volume (`180709_E`, 16.8M voxels, 59,408 vertices)

Extracted from [`workspace/reports/phase2_profiling/full_volume_edges_benchmark.json`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/workspace/reports/phase2_profiling/full_volume_edges_benchmark.json):

| Sub-Stage | Historical Windows (`v18`) | Initial macOS (`r2024b_v1`) | Phase 2 Optimized macOS | Total Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Watershed Discovery** | ~5,400 s (est.) | 748.77 s | **57.62 s** | **$13.00\times$ Faster** |
| **Edge Selection & Finalize** | ~134 s (est.) | 15.21 s | **15.53 s** | — |
| **Total Edges Stage** | **5,534.0 s** ($92.2\text{ min}$) | **764.0 s** ($12.7\text{ min}$) | **73.14 s** ($\mathbf{1.22\text{ min}}$) | **$10.45\times$ vs macOS<br/>$75.66\times$ vs Windows** |

```mermaid
graph LR
    subgraph Speedup["Edges Stage Wall-Clock Progression (Full Volume 180709_E)"]
        direction LR
        Win["Windows Baseline<br/>5,534s (92.2 min)"] --> MacInit["macOS Initial<br/>764s (12.7 min)"]
        MacInit --> MacOpt["macOS Phase 2 Optimized<br/>73.1s (1.2 min)"]
    end
```

#### Exact Parity Certification
- **Candidate Traces:** $84,650$ candidates materialized.
- **Connection Array:** $69,500 / 69,500$ connections.
- **Parity Assertion:** `np.array_equal(edges.connections, expected_connections) == True` (**100% BIT-IDENTICAL TO ORACLE**).

---

### 6.2 Energy Stage: 3D FFT & Pre-Indexed Symmetric IFFT Acceleration (ADR 0015)

The Energy stage consumed 91.3% of pipeline runtime ($9,342.1\text{ s}$). Under [ADR 0015](../adr/0015-phase2-energy-fft-preindexed-symmetric-ifft.md), we eliminated repeated 300+ MB temporary allocations and accelerated FFT execution via:
1. **Pre-Indexed Conjugate Symmetric IFFT:** Thread-safe `_SYMMETRIC_PARTNER_CACHE` pre-indexes reflection tuples $(iy, ix, iz)$ and $(py, px, pz)$ once per shape, avoiding dynamic indexing in every inverse FFT call.
2. **Analytic Self-Partner Direct Indexing:** Proved that self-partner frequencies under cyclic reflection ($lin == plin$) exist solely at DC ($0$) and Nyquist ($N/2$) frequencies ($\le 8$ voxels in any 3D grid). Replaced 16.8M boolean mask scans with direct analytic coordinate tuples.
3. **Thread-Aware `scipy.fft` Integration:** Multi-core SIMD-vectorized execution (`workers=-1` when serial, `workers=1` under Joblib parallel chunks).
4. **Chunk Engine Short-Circuiting:** Evaluates diagonal curvatures first and skips 6 inverse FFTs ($c_3, c_4, c_5$ cross curvatures and $g_0, g_1, g_2$ gradients) when a chunk contains no candidate voxels.

#### Micro-Benchmark Results (`(128, 128, 64)` chunk on Apple Silicon ARM64)

| FFT Operation | NumPy Single-Core | SciPy 1-Worker | SciPy All Cores | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Forward 3D FFT (`fftn`)** | 8.94 ms | 3.93 ms | **1.13 ms** | **$7.92\times$ Speedup** |
| **Inverse 3D FFT (`ifftn`)** | 8.70 ms | 5.78 ms | **1.51 ms** | **$5.75\times$ Speedup** |
| **Symmetric IFFT (`_ifftn_matlab_symmetric`)** | 22.64 ms (original) | 13.50 ms | **8.37 ms** | **$2.71\times$ Speedup** |

#### Parity Verification
- **Unit Suite:** 146 passed, 0 failed (`tests/unit/pipeline/energy/`).
- **Fixture Parity:** Stored MATLAB fixture `test_native_hessian_matlab_fixture.py` passes with `Max abs diff = 0.0`.

---

## 7. Verification & Reproducibility

- **Edges Profiling Source Code:** [`scripts/profiling/profile_edges_detailed.py`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/scripts/profiling/profile_edges_detailed.py)
- **Edges Full Volume Benchmark Script:** [`scripts/profiling/benchmark_full_edges.py`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/scripts/profiling/benchmark_full_edges.py)
- **Energy FFT Benchmark Script:** [`scripts/profiling/benchmark_energy_fft.py`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/scripts/profiling/benchmark_energy_fft.py)
- **Raw Profiling Dumps:**
  - Crop Profiling Summary: [`workspace/reports/phase2_profiling/edges_detailed_profile.json`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/workspace/reports/phase2_profiling/edges_detailed_profile.json)
  - Full Volume Benchmark Summary: [`workspace/reports/phase2_profiling/full_volume_edges_benchmark.json`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/workspace/reports/phase2_profiling/full_volume_edges_benchmark.json)
- **Reproduce anytime via:**
  ```bash
  # Crop Harness detailed Edges profiling
  uv run python scripts/profiling/profile_edges_detailed.py

  # Full Volume Edges benchmark & bit-identical parity check
  uv run python scripts/profiling/benchmark_full_edges.py

  # Energy FFT & Symmetric IFFT Acceleration benchmark
  uv run python scripts/profiling/benchmark_energy_fft.py --iterations 20 --n-jobs 1
  ```
