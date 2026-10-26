# MATLAB to Python Scientific Translation: A Study in Exact Parity

*Draft generated for journal submission (target: JORS / SoftwareX / IEEE CiSE).*

> **Methodology & publication draft — not live status.**
>
> **Status context:** Phase 1 exact-route parity is **CLOSED** and certified on destination `canonical_full_v18` under ADR 0011 (Energy, Vertices) and ADR 0012 (Edges, Network), with baseline frozen in [phase1-baseline-freeze.json](../reference/core/phase1-baseline-freeze.json) and profiling baseline in [phase2-profiling-baseline.json](../reference/core/phase2-profiling-baseline.json). Live pass/fail: [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

## Abstract

Translating legacy scientific computing pipelines from proprietary, array-centric environments (MATLAB) to open-source, modular languages (Python) frequently encounters subtle divergences in floating-point arithmetic, array memory layout, and tie-breaking heuristics. In high-stakes biomedical image analysis, such as 3D vascular network vectorization from multi-scale Hessian-filtered volumetric optical scans [1–3, 7], minor numerical drift compounds nonlinearly across pipeline stages, ultimately altering graph topology and clinical morphometry. This paper presents the design, formal mathematical principles, and software architecture of the SLAVV translation framework, achieving certified 1:1 structural and topological parity against legacy MATLAB oracles on large volumes ($512 \times 512 \times 64$ voxels). We formalize the exact discrete-continuous boundary certification policy (ADR 0011/0012), dissect the memory paradigm clash between Fortran column-major and C row-major ordering, and present ten certified, parity-preserving algorithmic innovations—spanning batched Hessian eigensolvers, sparse IFFT conjugate-symmetry, and JIT-compiled watershed geometric propagation [4–6, 8, 13]—that enable multi-gigabyte volume vectorization on standard workstation hardware.

## 1. Introduction & Motivation

Moving a scientific computing pipeline from MATLAB to Python represents a shift from a research-first script environment to a scalable, production-grade application. Our focus has been on translating the SLAVV (segmentation-less automated vascular vectorization) pipeline [1].

A critical challenge in this migration is ensuring mathematical trust. Instead of targeting "good enough" statistical correlation, our Phase 1 milestone achieved **1:1 structural exact parity**: zero missing and zero extra vertices or edges between the legacy MATLAB oracle and the new Python output across the canonical volume ($V_{\mathrm{canonical}} = 512 \times 512 \times 64$).

This paper serves as both a retrospective of the translational challenges faced and a repository of technical lessons that act as soil for Phase 2 optimizations. The ten parity-preserving speed and memory improvements that made that certification runnable on a workstation are cataloged in [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md); the sections below formalize the methodology and detail the translation architecture.

## 2. Phase 1: The Exact Parity Method & Mathematical Formalism

Exact parity provides a high-trust guarantee: deviations on discrete or topological structure signal a logic divergence, while bounded float differences reflect numerical rather than logical disagreement.

### 2.1 The Discrete-Continuous Certification Boundary (ADR 0011)

The method began as a zero-tolerance gate (`np.equal`, zero ULP). Cross-library BLAS/LAPACK [13] and FFT non-associativity [10, 14, 15] impose an irreducible numerical floor near $\epsilon_{\mathrm{BLAS}} \approx 2\times 10^{-11}$ on continuous fields. In alignment with established golden-master numerical porting methodologies (such as the CARFAC v2 reference port [12]), the framework establishes a formal bifurcation:

$$\mathcal{D}_{\mathrm{discrete}} = \{ (y, x, z) \in \mathbb{Z}^3, \, s \in \mathbb{N}, \, \mathcal{C} \in \mathbb{N}^{E \times 2} \} \implies \text{Strict Equality } (\Delta = 0)$$

$$\mathcal{F}_{\mathrm{continuous}} = \{ E(y, x, z) \in \mathbb{R}, \, r \in \mathbb{R}^+ \} \implies |E_{\mathrm{py}} - E_{\mathrm{mat}}| \le \text{atol} + \text{rtol} \cdot |E_{\mathrm{mat}}|$$

with $\text{rtol} = 10^{-7}$ and $\text{atol} = 10^{-9}$. Discrete indices (seed voxel locations, scale labels, vertex-to-vertex connections, and multiset strand counts) must match MATLAB identically with zero tolerance, while continuous fields (principal eigenvalues, local energy values, and Euclidean path lengths) satisfy strict relative tolerances.

### 2.2 Dual-Surface Energy Ranking Model (ADR 0013)

A core challenge in edge discovery is the divergence between the static Hessian energy field $E_0(y, x, z)$ and the dynamic watershed-mutated claimed energy map $E_{\mathrm{claimed}}(y, x, z)$ [11]. During watershed flood-fill, voxels claimed by a vertex $v$ are penalized via scale tolerance, geometric distance, and directional suppression weights:

$$E_{\mathrm{claimed}}(p) = E_0(p) \cdot \exp\left(-\frac{1}{2} \left(\frac{\Delta s}{\tau_s}\right)^2\right) \cdot \frac{1 - \cos(\pi \cdot \min(1, \frac{4}{3} r/R))}{2} \cdot \exp\left(-\frac{1}{2} \left(\frac{3 \cdot d/r}{\tau_d}\right)^2\right) \cdot \max(0, \hat{u} \cdot \hat{v})$$

When candidate edges $\mathcal{E}$ are ranked during Edge Selection, candidate traces carry the claimed surface samples:

$$\mathrm{Rank}(e_{ij}) = \max_{p \in \mathrm{Trace}(e_{ij})} E_{\mathrm{claimed}}(p)$$

Baking claimed trace provenance at Watershed Discovery finalize guarantees that Python's raw-max `sort_edges` selection produces identical graph topology to MATLAB's mutated array.

### 2.3 Emergent Watershed Topology & Spatial-Multiset Parity (ADR 0012)

Because the global watershed flood-fill is a greedy, shared-state competitive process across active catchments, tiny variations in queue popping order cause chaotic divergences in raw edge emission sequences. Grounded in medical image validation theory [9], the framework formalizes certification via spatial catchment overlap and topological graph multiset isomorphism rather than brittle pixel/pair-order equivalence:

1. **Voxel-Ownership Agreement Metric (Spatial Catchment Parity):** Across the active foreground domain $\Omega_{\mathrm{claimed}} = \{ p \in \Omega \mid \mathcal{M}_{\mathrm{mat}}(p) \notin \{ \text{background}, \text{border} \} \}$:
   $$\mathcal{A}_{\mathrm{ownership}} = \frac{1}{|\Omega_{\mathrm{claimed}}|} \sum_{p \in \Omega_{\mathrm{claimed}}} \mathbf{1}\left( \mathcal{M}_{\mathrm{py}}(p) = \mathcal{M}_{\mathrm{mat}}(p) \right) \ge \tau_{\mathrm{ownership}}$$
   where the certification threshold $\tau_{\mathrm{ownership}} \ge 60\%$ is far exceeded by the production system ($99.99986\%$ on $V_{\mathrm{canonical}}$).

2. **Order-Independent Graph Multiset Isomorphism:** The final vascular network is certified by exact multiset equivalence over structural graph invariants:
   $$\mathcal{M}(\mathcal{S}_{\mathrm{py}}) = \mathcal{M}(\mathcal{S}_{\mathrm{mat}}), \quad \mathcal{M}(\mathcal{B}_{\mathrm{py}}) = \mathcal{M}(\mathcal{B}_{\mathrm{mat}})$$
   where $\mathcal{S}$ represents strand endpoint pairs $(v_{\mathrm{start}}, v_{\mathrm{end}})$ and $\mathcal{B} = \{ v \in \mathcal{V} \mid \operatorname{deg}(v) \ge 3 \}$ represents bifurcation nodes.

3. **Sub-Voxel Continuous Strand Geometry:** Matched strand centerlines $\mathbf{x}(t)$ and energy traces $E(t)$ are evaluated under continuous tolerances:
   $$\|\mathbf{x}_{\mathrm{py}}(t) - \mathbf{x}_{\mathrm{mat}}(t)\|_\infty \le \epsilon_{\mathrm{geom}}, \quad |E_{\mathrm{py}}(t) - E_{\mathrm{mat}}(t)| \le \text{atol} + \text{rtol} \cdot |E_{\mathrm{mat}}(t)|$$

## 3. Floating Point & Numerical Nuances

MATLAB and Python (NumPy/SciPy [4, 5]) implement mathematical primitives differently, and those differences compound nonlinearly through multi-stage pipelines.

* **Precision Collapsing**: MATLAB defaults to IEEE-754 `double` (64-bit) [15]. Downcasting intermediate watershed maps or vertex energies to `float32` collapsed adjacent floating-point values into single bins, altering seed priority in dense microvascular hubs.
* **Mesh & Interpolation Discrepancies**: MATLAB `linspace(a, b, N)` computes fractional increments that produce infinitesimal drift (e.g. $9.000000000000002$) feeding `interp3`. NumPy `linspace` or arithmetic strides yield distinct rounding trajectories; Python reproduces MATLAB-specific linspace calculations to prevent branch divergences during trilinear interpolation.
* **Boundary Inf Propagation**: MATLAB `interp3` propagates $\infty$ boundary conditions across all immediate neighbors, suppressing spurious candidate seeds along volume edges. Standard SciPy interpolators (`scipy.interpolate.interpn` [5]) discard or extrapolate edge values, requiring specialized interpolation kernels in Python (`_interp3_matlab_linear_inf`).
* **Fortran Lowest Linear Index Tie-Breaking**: When two voxels have identical continuous energy $E(p_1) = E(p_2)$, MATLAB resolves the tie by choosing the voxel with the lowest Fortran-order linear index:
$$\mathrm{Priority}(p) = \arg\min \left( E(p), \, \mathrm{LinIdx}_{\mathrm{Fortran}}(p) \right)$$
Python operates in C-order by default [4]. To preserve deterministic hub expansion, the Exact Route enforces an internal $[Y, X, Z]$ grid with Fortran (F) memory order.

### 3.1 Micro-Architectural Reverse Engineering of Closed Built-Ins

While user-facing MATLAB scripts are distributed as open `.m` files, the performance-critical inner loops of the MATLAB Image Processing Toolbox compile to closed-source, platform-native dynamic libraries (`libmwimagescore.dylib`, `libmwmorphop_flat.dylib`, and `libmwimfilter.dylib`). Discrepancies between standard Python libraries (such as `scipy.ndimage` or `scikit-image`) and MATLAB frequently stem from unpublicized algorithmic choices within these compiled binaries. Using the Reverse Engineering Anything (REA) framework and native ARM64 disassembly (`llvm-objdump`), we conducted an instruction-level audit of the four core built-ins underpinning the vectorization pipeline:

1. **Watershed Segmentation (`libmwimagescore.dylib` / `_watershed_meyer`, 13,114 instructions):**
   Disassembly reveals that MATLAB implements Fernand Meyer’s topographic watershed [11] via a hierarchical array of bucket priority queues rather than a standard binary heap. Active boundary voxels are inserted into linked lists indexed by discretized energy levels. Crucially, within identical energy buckets, insertion order and tie-resolution strictly adhere to the Fortran column-major linear index:
   $$\mathrm{LinIdx}(y, x, z) = y + x \cdot N_y + z \cdot N_y N_x$$
   This low-level invariant explains why arbitrary C-contiguous priority queues produce divergence in dense microvascular hubs, providing the micro-architectural justification for the composite tie-break keys in Innovation 3.3.

2. **Distance Transforms (`libmwimagescore.dylib` / `_graydist`, 2,962 instructions):**
   The native `_graydist` implementation utilizes an exact integer squared Euclidean distance accumulator (`_bwdistComputeEDT`) combined with Intel TBB / Grand Central Dispatch parallelization. Geodesic distance updates enforce an asymmetric chamfer metric where diagonal steps in 3D scale with strict float64 multipliers ($\sqrt{1}, \sqrt{2}, \sqrt{3}$), preventing rounding collapse observed in 32-bit approximations.

3. **Flat 3D Morphology (`libmwmorphop_flat.dylib` / `is_linear_strel`, 81 instructions; `dilate_flat_real64`, 2,240 instructions):**
   Disassembly of `is_linear_strel` reveals an internal fast-path optimizer that detects 1D collinear structuring element components. When a structuring element decomposes into linear segments, MATLAB evaluates multi-directional morphology via cascaded 1D sliding-window min/max filters initialized to IEEE-754 $\pm\infty$ [15] outside boundary domains. This architecture guarantees that flat dilations on float arrays do not clip sub-zero minimums, validating Python's structuring-element cache and boundary-padding conditions.

4. **Spatial Multidimensional Filtering (`libmwimfilter.dylib` / `imfilter_real64`, 1,100 instructions):**
   The compiled filter engine operates strictly with 64-bit double-precision accumulators and supports cyclic index reflections via `update_linear_index`. This confirms that intermediate convolution rounding stays bounded within machine epsilon ($\epsilon_{\mathrm{mach}} \approx 2.22 \times 10^{-16}$), eliminating numerical drift during 3D Gaussian gradient and Hessian tensor filtering.

The full Ground-Truth Architecture Specifications extracted from these compiled binaries are published in `docs/reference/reverse_engineering/`.

## 4. Architectural Translation & State Integrity

Phase 1 preserved mathematical structure while replacing MATLAB’s unstructured workspace injection with typed Python ownership. `RunContext`, `StageController`, and explicit parameter validation now prevent silent mutation of run state. Pass-through facades were mapped rather than deleted, which keeps pipeline depth intact and leaves a clear seam for Phase 2 interface simplification.

## 5. Scaling & Performance Breakthroughs

Translating a pipeline for certified parity often means keeping memory-intensive intermediate structure that MATLAB could afford on a single large array. Scaling to full canonical volumes ($512\times 512\times 64$ and beyond) exposed Energy and Edges bottlenecks that would not fit on a 16 GB workstation. Ten parity-preserved innovations made that volume runnable without changing certified numerical behavior. Mechanism, complexity, and verification for each item live in [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md).

### 5.1 In-place octave accumulation

Early Python Energy matched MATLAB’s batch style by allocating 4D intermediates $NY\times NX\times NZ\times S$ per octave. Parallel workers then hit `ArrayMemoryError`. `matlab_get_energy_v202_chunked.py` now accumulates the per-scale minimum in place, $\mathrm{Accumulator}\leftarrow\min(\mathrm{Accumulator},E_{\mathrm{scale}})$, and tracks argmin in an `int16` buffer. Catalog [Innovation 1.1](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-11-4d-multi-scale-scale-stack-elimination-in-place-octave-accumulation) reports the measured drop from about 300 MiB/thread to about 10 MiB/thread while keeping bit-identical minima.

### 5.2 Batched Hessian eigensolver

MATLAB `energy_filter_V200.m` builds a cell array of $3\times 3$ Hessians and calls `cellfun(@eig, ...)`. Python reconstructs a batched `(B, 3, 3)` tensor and uses LAPACK `np.linalg.eigh` [4, 13] plus `np.einsum` for gradient projection. Catalog [Innovation 1.2](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-12-batched-3-times-3-hessian-eigendecomposition--einstein-summation) is the record for that substitution; Energy floats still certify under ADR 0011 `allclose`, not last-digit stretch.

### 5.3 Deterministic chunk parallelism

MATLAB `parfor` merged chunks by disk completion order. Python Joblib workers remain pure and merge strictly in `chunk_idx` order so `n_jobs>1` stays bit-exact with serial. Catalog [Innovation 1.3](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-13-deterministic-parallel-chunking-with-fixed-reduction-ordering) and the operator note in [exact-energy-chunk-parallelism.md](../solutions/parity/exact-energy-chunk-parallelism.md) document the `--n-jobs auto` opt-in; dest default remains serial.

### 5.4 Vertex painting structuring-element cache

`paint_vertex_image` used to rebuild an `ellipsoid()` mask for every vertex. The occupancy offsets now memoize on the discrete radius tuple $(r_y,r_x,r_z)$, so mesh generation is $O(S)$ unique scales rather than $O(V)$ vertices. Catalog [Innovation 2.1](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-21-scale-structuring-element-offset-caching-for-volume-painting) records that claim; occupancy identity is the unit-test surface for the cache.

### 5.5 Sparse conjugate-symmetry IFFT

After 4D elimination, high-resolution chunks still duplicated full conjugate-mirror grids inside MATLAB-style symmetric IFFT. `_ifftn_matlab_symmetric` now updates conjugate pairs in place from a sparse linear-index mask where $lin>plin$. Catalog [Innovation 3.1](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-31-sparse-conjugate-symmetry-for-memory-safe-3d-ffts) reports the halved FFT working set.

### 5.6 Claimed Trace Energy bake

MATLAB ranks candidates from the watershed-mutated energy map [11]. Python now bakes that claimed surface onto each candidate at Watershed Discovery finalize ([ADR 0013](../adr/0013-claimed-energy-trace-provenance.md)), so Edge Selection’s raw-max `sort_edges` sees the same ranking input. Catalog [Innovation 3.2](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-32-claimed-trace-energy-baked-provenance-adr-0013) is the mechanism note; live Edges/Network standing is ONE TRUTH, not this draft.

### 5.7 Indexed watershed heap

MATLAB frontier maintenance copied sorted arrays (`[front; new_loc; back]`) at every voxel. Python uses an indexed binary heap keyed by `(Energy, OriginSeedRank, FortranLinearIndex)` so push/pop is $O(\log N)$ and ties still follow lowest Fortran index. Catalog [Innovation 3.3](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-33-deterministic-priority-queue-tie-breaking-olog-n-acceleration) is the complexity record for that queue.

### 5.8 Sparse CSR strand decomposition

MATLAB network assembly walked cell arrays with nested `find`. Python builds a CSR adjacency matrix and classifies interior versus bifurcation vertices from degree, then takes connected components of the interior subgraph (`scipy.sparse.csgraph` [5]). Catalog [Innovation 4.1](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-41-sparse-csr-graph-decomposition-for-vascular-strands) records the graph rewrite.

### 5.9 Arc-length centerline resampling

MATLAB smoothed discrete pixel polylines with 2D Gaussian kernels. Python parameterizes traces by cumulative Euclidean arc length, applies a 1D energy-weighted Gaussian, and interpolates to uniform samples. Catalog [Innovation 4.2](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-42-continuous-arc-length-centerline-smoothing--interpolation) is the smoothing note.

### 5.10 JIT-compiled geometric penalties & voxel claiming

The global watershed discovery loop evaluates neighborhood structuring elements (strels) across millions of steps. Naive Python implementations incur severe memory overhead from repeated NumPy array allocations for size penalties, distance adjustments, directional projections, and multi-map updates. Python compiles the geometric calculations and atomic voxel claiming loops via Numba JIT [8] while maintaining bit-identical topological invariants and safe pure-Python fallbacks. Catalog [Innovation 3.4](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md#innovation-34-jit-compiled-strel-geometric-penalties--atomic-voxel-claiming-numba) documents the verified parity and mechanism.

## 6. Performance Innovations Catalog

The companion catalog is the table of record for MATLAB bottleneck, Python substitution, parity check, and reported speed or memory gain:

[Parity-Preserved Performance Innovations Catalog](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md)

Do not copy those measured figures into HANDOFF, TODO, or ONE TRUTH.

### 6.1 Cross-Version & Cross-Architecture Generalization (R2019a/x86_64 to R2024b/ARM64)

A critical question in scientific translation is whether exact parity is fragile to environment drift—such as compiler upgrades, newer runtime math libraries, or architecture transitions. While the frozen Phase 1 baseline (`canonical_full_v18`) certified against MATLAB R2019a/R2024a on x86_64 Windows, we independently replicated the full canonical pipeline against **MATLAB R2024b on ARM64 Apple Silicon** (`canonical_full_r2024b_v1`):

1. **Discrete & Topological Preservation:**
   - **Energy:** Evaluated under ADR 0011 across all 16,777,216 voxels, yielding **0 scale mismatches** (100.0% discrete scale agreement) and maximum continuous deviation bounded at $2.72 \times 10^{-7}$.
   - **Vertices:** 100% exact match across all $59,408$ vertex positions, scales, and rank-ramp energies.
   - **Edges:** Reached exact array identity over the entire connection set:
     $$\texttt{np.array\_equal}(\mathcal{C}_{\mathrm{py}}, \mathcal{C}_{\mathrm{mat}}) = \mathrm{True} \quad (69,500 \text{ identical pairs in exact order})$$
   - **Network:** 100% bifurcation identity ($25,371$ nodes, `np.array_equal == True`) and exact strand multiset match ($48,049$ pairs, ADR 0012 passed).

2. **Hardware Acceleration on Apple Silicon:**
   - Utilizing 6 worker threads with Joblib parallel chunking on Apple Silicon (M-series, unified memory), steady-state throughput reached **$2.59\text{ s/chunk}$** on Octave 1 (down from the $44.4\text{ s/chunk}$ serial baseline), demonstrating a **$17.1\times$ wall-clock speedup** without a single bit of topological drift.
   - Total full-volume pipeline runtime dropped to $\approx 2.9\text{ hours}$ (compared to $>50\text{ hours}$ in unoptimized serial execution).

This confirms that the mathematical abstractions, tie-breaking heuristics, and in-place accumulators engineered into the Python implementation form a durable, architecture-independent specification for vascular vectorization.

## 7. Phase 2 Ideation & Profiling Baseline

With Phase 1 certified and frozen on `canonical_full_v18` ([phase1-baseline-freeze.json](../reference/core/phase1-baseline-freeze.json)), the codebase is a testbed for further speedup. Read-only timings in [phase2-profiling-baseline.json](../reference/core/phase2-profiling-baseline.json) identify Edges (5,534 s, about 92.2 min) as the dest bottleneck, with Network at 416 s. Remaining ideas are multi-core or JIT acceleration of the watershed frontier, GPU/CuPy for multi-scale 3D FFTs and Hessian filtering, and C-order memory unwind only behind an explicit Phase 2 ADR. None of those reopen Phase 1.

## 8. The Unexpected Consequences of Exact Parity

Pursuing bit-perfect MATLAB equivalence shifts the work from “translating logic” toward “building a MATLAB emulator in Python.” The goal is mathematical confidence for certification. The cost is architectural.

### 8.1 The Memory Paradigm Clash (Fortran vs. C-Order)

MATLAB is column-major and thinks in `[Y, X, Z]`. NumPy is row-major and natively thinks in `[Z, Y, X]` [4]. When two voxels share an energy, Python must pick the same lowest linear index MATLAB would, so Exact Route transposes into `[Y, X, Z]`, processes Fortran-contiguous blocks, and transposes back. That costs extra copies and cache misses, which is why the batched eigensolver and in-place accumulators exist.

### 8.2 The Rejection of the Python Data Science Ecosystem

A typical Python port would call `scipy.ndimage` [5] or `skimage` [6]. Those libraries do not match MATLAB on the cases that move vertices and edges: `interpn` versus `interp3` on `Inf`, banker’s rounding versus MATLAB round-half-up, and `linspace` drift. The project therefore keeps shims such as `_interp3_matlab_linear_inf`, `_matlab_zero_based_linspace`, and `np.floor(x + 0.5)` inside `_matlab_uint16_cast`.

### 8.3 "Bug-for-Bug" Compatibility

Zero missing and extra edges means Python must reproduce MATLAB quirks, not repair them. `float32` collapsing bits is the canonical example: it is not a better float, it is a different tie-break. The Python tree therefore inherits MATLAB technical debt that later readers will find only in the oracle contract.

### 8.4 The Massive Overhead of the Proof Harness

Unit tests cannot certify billions of voxels. The exact-proof coordinator, parameter-diffusion hashes, cheap parity ladder, and crop-then-canonical pre-gate are as large as the pipeline they guard. A parity-sensitive change still requires evidence in the form specified by [PARITY_RUN_EVIDENCE.md](../reference/workflow/PARITY_RUN_EVIDENCE.md), not a green unit file alone.

---

## 9. References

1. **Mihelic, S. A., et al. (2021).** "Segmentation-Less, Automated, Vascular Vectorization." *PLOS Computational Biology*, 17(10), e1009451. [DOI: 10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451) | [OpenAlex: W3201353661](https://openalex.org/W3201353661)
2. **Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998).** "Multiscale vessel enhancement filtering." *MICCAI*, 130–137. [DOI: 10.1007/BFb0056195](https://doi.org/10.1007/BFb0056195) | [OpenAlex: W2129534965](https://openalex.org/W2129534965)
3. **Sato, Y., et al. (1998).** "Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images." *Medical Image Analysis*, 2(2), 143–168. [DOI: 10.1016/S1361-8415(98)80009-1](https://doi.org/10.1016/S1361-8415(98)80009-1) | [OpenAlex: W2096320880](https://openalex.org/W2096320880)
4. **Harris, C. R., et al. (2020).** "Array programming with NumPy." *Nature*, 585(7825), 357–362. [DOI: 10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) | [OpenAlex: W3035965352](https://openalex.org/W3035965352)
5. **Virtanen, P., et al. (2020).** "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods*, 17(3), 261–272. [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) | [OpenAlex: W3003257820](https://openalex.org/W3003257820)
6. **van der Walt, S., et al. (2014).** "scikit-image: image processing in Python." *PeerJ*, 2, e453. [DOI: 10.7717/peerj.453](https://doi.org/10.7717/peerj.453) | [OpenAlex: W2015159529](https://openalex.org/W2015159529)
7. **Blinder, P., et al. (2014).** "Vascular Supply of the Cerebral Cortex is Specialized for Cell Layers but Not Columns." *Cerebral Cortex*, 24(8), 2073–2085. [DOI: 10.1093/cercor/bhu221](https://doi.org/10.1093/cercor/bhu221) | [OpenAlex: W2041343897](https://openalex.org/W2041343897)
8. **Lam, S. K., Pitrou, A., & Seibert, S. (2015).** "Numba: A LLVM-based Python JIT compiler." *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC (LLVM '15)*, 1–6. [DOI: 10.1145/2833157.2833162](https://doi.org/10.1145/2833157.2833162) | [OpenAlex: W2245493112](https://openalex.org/W2245493112)
9. **Müller, D., et al. (2022).** "Towards a guideline for evaluation metrics in medical image segmentation." *BMC Research Notes*, 15(1), 210. [DOI: 10.1186/s13104-022-06096-y](https://doi.org/10.1186/s13104-022-06096-y) | [PMC: PMC9208116](https://pmc.ncbi.nlm.nih.gov/articles/PMC9208116/) | [OpenAlex: W4283160212](https://openalex.org/W4283160212)
10. **Ahrens, W., Demmel, J., & Nguyen, H. D. (2020).** "Algorithms for Efficient, Reproducible Floating-Point Summation." *ACM Transactions on Mathematical Software*, 46(3), 1–35. [DOI: 10.1145/3389360](https://doi.org/10.1145/3389360) | [OpenAlex: W3041643290](https://openalex.org/W3041643290)
11. **Meyer, F. (1994).** "Topographic distance and watershed lines." *Signal Processing*, 38(1), 113–125. [DOI: 10.1016/0165-1684(94)90060-4](https://doi.org/10.1016/0165-1684(94)90060-4) | [OpenAlex: W2025818287](https://openalex.org/W2025818287)
12. **Lyon, R. F., et al. (2024).** "The CARFAC v2 Cochlear Model in Matlab, NumPy, and JAX." *arXiv preprint arXiv:2404.17490*. [DOI: 10.48550/arXiv.2404.17490](https://doi.org/10.48550/arXiv.2404.17490) | [OpenAlex: W4396243990](https://openalex.org/W4396243990)
13. **Anderson, E., et al. (1999).** "LAPACK Users' Guide." *Society for Industrial and Applied Mathematics (SIAM)*, 3rd ed. [DOI: 10.1137/1.9780898719604](https://doi.org/10.1137/1.9780898719604) | [OpenAlex: W1480928214](https://openalex.org/W1480928214)
14. **Higham, N. J. (2002).** "Accuracy and Stability of Numerical Algorithms." *Society for Industrial and Applied Mathematics (SIAM)*, 2nd ed. [DOI: 10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027) | [OpenAlex: W2020804487](https://openalex.org/W2020804487)
15. **IEEE Standards Association (2019).** "IEEE Standard for Floating-Point Arithmetic." *IEEE Std 754-2019*, 1–84. [DOI: 10.1109/IEEESTD.2019.8766229](https://doi.org/10.1109/IEEESTD.2019.8766229) | [OpenAlex: W4233996382](https://openalex.org/W4233996382)

---
*End of Draft. Maintained for publication submission.*
