# ADR 0014: Phase 2 Performance Optimization and Parity Regression Gates

## In short

Phase 2 transitions the SLAVV codebase from exact MATLAB emulation to micro-architectural optimization. Every optimization to Watershed Discovery queue operations, structuring element lookups, and memory layout must pass strict regression gates—preserving 100% bit-identical candidate counts, vertex connectivity, and edge connections against the frozen baseline before promotion. Historical Phase 1 baselines (`canonical_full_v18` and `canonical_full_r2024b_v1`) remain immutable and protected.

## Status

Accepted (2026-10-26)

## Context

Phase 1 achieved exact mathematical parity across all four pipeline stages (ADR 0011 for Energy, ADR 0012 for Edges and Network, ADR 0013 for Claimed Trace Provenance), proving that Python reproduces MATLAB R2019a and R2024b bit-for-bit on both Windows x86_64 and Apple Silicon ARM64.

With mathematical trust certified and the Phase 1 canonical baselines frozen, Phase 2 focuses on throughput, scalability, and micro-architectural optimization.

Comprehensive profiling ([`docs/investigations/PHASE2_PERFORMANCE_PROFILING_REPORT.md`](../investigations/PHASE2_PERFORMANCE_PROFILING_REPORT.md)) established that:
1. **Watershed Candidate Discovery** accounts for **98.01%** of the Edges stage wall-clock time (748.8s out of 764.0s on the canonical full volume).
2. Within Watershed Discovery, **69.32%** of execution time is consumed by queue maintenance:
   - $O(N)$ linear scans in `_matlab_global_watershed_insert_available_location`: **39.06%**.
   - Repeated full-list scans and slice allocations in `_matlab_global_watershed_reset_join_locations`: **30.26%**.
3. Repeated structuring element geometry rebuilding accounts for **15.66%** of runtime.
4. Mathematical floating-point calculations (Numba directional penalties and atomic voxel claiming) are already optimized, consuming $<4\%$ of total discovery time.

Without strict regression gates, aggressive optimization could subtly alter edge traversal order or tie-breaking semantics, compromising the certified exact parity achieved in Phase 1.

## Decision

We establish five mandatory rules for all Phase 2 optimizations:

### 1. Bit-Identical Regression Gate
Any optimization to `SortedFrontier`, queue data structures, or structuring element caches must produce **100% bit-identical** outputs against the verified reference oracle on both crop and full-volume datasets:
- **Crop Verification:** Exact candidate count ($19,225$ candidates on `crop_M_r2024b`), bit-identical candidate trace coordinates, and identical final edge connections ($15,511$ edges).
- **Full Volume Verification:** Exact connection array equality (`np.array_equal == True`, $69,500 / 69,500$ connections on `180709_E`), zero vertex mismatches, and passing ADR 0012 multiset network topology parity.

### 2. Strict Tie-Breaking Preservation in Frontier Queues
MATLAB's frontier queue exhibits asymmetric tie-breaking semantics between seed phases:
- **`seed_idx == 1`:** When candidate energies match, the new location is inserted **before** equal elements (further from the tail, lower pop priority / FIFO-like).
- **`seed_idx != 1`:** When candidate energies match, the new location is inserted **after** equal elements (closer to the tail, higher pop priority / LIFO-like).

Any accelerated frontier queue (such as binary search via `bisect`, balanced binary search trees, or B-trees) must preserve this exact tie-breaking behavior. Standard min-heaps (`heapq`) that break ties arbitrarily or purely on insertion order are prohibited for production watershed discovery unless augmented with full MATLAB tie-break keys.

### 3. Protection of Frozen Baseline Roots
Historical claim and validation run roots are immutable:
- `workspace/runs/oracle_180709_E/canonical_full_v18`
- `workspace/runs/oracle_180709_E/canonical_full_r2024b_v1`
- `workspace/runs/oracle_180709_E/crop_M_exact_v3`
- `workspace/runs/oracle_180709_E/crop_M_stretch_engine_v2`

Active writers must never write into or mutate these directories. Phase 2 benchmark runs must target distinct run directories with explicit writer leases and manifests.

### 4. Seam Decoupling and Equivalence Suites
All algorithmic queue optimizations must:
1. Reside behind clean strategy seams (e.g. selectable via `build_watershed_frontier(backend=...)` or drop-in class interfaces).
2. Include comprehensive equivalence unit tests verifying identical behavior against the reference implementation across thousands of randomized energy configurations and edge cases (including $\pm\infty$ and duplicate energies).

### 5. Memory Layout Policy ("The Great Transpose")
Any future unwinding of Fortran column-major order to native C-contiguous arrays must be executed under a separate, dedicated Phase 2 ADR. Algorithmic queue acceleration ($O(\log N)$ bisect and join reset optimization) and structuring element memoization must be completed and validated first, decoupling data structure speedups from spatial indexing changes.

## Consequences

- Watershed Candidate Discovery can be accelerated by an estimated $3\times - 5\times$, reducing Edges stage wall-clock time from 12.7 minutes to $<3$ minutes without risking parity drift.
- Developers and agents have clear, non-negotiable pass/fail criteria for PRs touching the core discovery loop.
- The Cheap Parity Ladder remains the enforcement mechanism: unit equivalence tests $\to$ crop bit-identical verification $\to$ full volume certification.

## Considered Options

| Option | Why rejected for production |
| :--- | :--- |
| **Standard Min-Heap (`heapq`)** | Does not preserve MATLAB's asymmetric `seed_idx` tie-breaking semantics; causes alternate edge selection paths and graph topological divergence in dense capillary clusters. |
| **Immediate Full C-Order Rewrite** | High complexity and high risk of subtle coordinate transpositions across 4 pipeline stages simultaneously. Premature before algorithmic queue bottlenecks are eliminated. |
| **Approximate Watershed Clustering** | Violates Phase 1 exact parity guarantee; changes physical vessel topology and strand parameterization. |
| **Strict Bit-Identical Multi-Tier Regression Gate (Selected)** | Guarantees zero parity drift while unlocking maximum performance gains from algorithmic queue and cache optimizations. |
