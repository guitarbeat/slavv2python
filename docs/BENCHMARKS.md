# Benchmarks and Performance Improvements

This document outlines potential improvements for the Python port of SLAVV, comparing the current implementation with the original MATLAB approach and suggesting optimizations for speed and accuracy.

## Summary Comparison Table

| Step | Component | Original MATLAB (Per Mapping) | Current Python (`slavv2python`) | Proposed Improvement | Expected Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Energy Filter** | `imgaussfilt3` (Hybrid Spatial/Freq) | `ndimage.gaussian_filter` (Spatial) | **FFT Convolution** for $\sigma > 10$ | **10x speedup** on large scales |
| **1** | **Hessian Eigenvalues** | Vectorized Linear Algebra | `skimage` loop/wrapper | **Numba / GPU (CuPy)** | **5-10x speedup** |
| **2** | **Vertex Extraction** | `imregionalmin` (Discrete) | `minimum_filter` (Discrete) | **Peak Finding optimizations** | Minor |
| **3** | **Edge Tracing** | `while` loops (JIT Compiled) | `while` loops (Interpreter) | **Numba JIT (`@njit`)** | **100x speedup** (Critical) |
| **3** | **Trace Method** | Voxel-Snapped (Discrete) | Continuous (Float) | **Option: `discrete_tracing=True`** | **2x speedup** (Trade-off) |
| **3** | **Alternative Algo** | Watershed (supported) | Watershed (functional) | **Switch to Watershed** | **20x speedup** vs Tracing |

---

## Detailed Recommendations based on `MIGRATION_GUIDE.md`

### verification of Mapping Claims
The mapping document (`docs/MIGRATION_GUIDE.md`) confirms that the Python port prioritizes **feature parity** and **accuracy** (e.g., "Continuous" tracing by default) over raw speed parity with MATLAB's JIT.

### 1. Energy Calculation: FFT vs. Spatial Convolution

**Observation:**
The mapping notes that Python kernel details are "simplified". MATLAB's `imgaussfilt3` is highly optimized and switches to FFT for large kernels. Python's `scipy.ndimage` remains spatial.

**Improvement:**
Implement smart switching:
- If $\sigma < 10$: Keep `gaussian_filter`.
- If $\sigma \ge 10$: Use FFT convolution (`scipy.signal.fftconvolve`). Complexity becomes $O(N \log N)$, independent of $\sigma$.

### 2. Edge Extraction: JIT and Discrete Tracing

**Observation:**
The mapping explicitly states: *"Tracing uses floating-point updates by default; enable `discrete_tracing=True` for voxel-snapped steps."*
Continuous tracing is numerically strictly superior (smoother vessels) but computationally much heavier for the Python interpreter.

**Improvement 1 (Algorithm):**
Users needing speed should set `discrete_tracing=True` in their parameters. This aligns closer to MATLAB's `get_edges_V300.m` numerical behavior and is faster to compute.

**Improvement 2 (Implementation):**
Use **Numba** to compile the `_trace_edge` loop. Since this function is purely mathematical, it is a perfect candidate for `@njit`.

### 3. Alternative: Watershed Segmentation

**Observation:**
The mapping lists `get_edges_by_watershed.m` as equivalent to `extract_edges_watershed`.

**Improvement:**
For very large datasets where tracing is too slow, users should switch `edge_method='watershed'`. This replaces the expensive "random walker" tracing with a global region-growing operation, which acts on the whole volume simultaneously (vectorized).

### 3. Parallelization Strategy

**Current State:**
The code has a chunking mechanism (`get_chunking_lattice`) but it processes chunks serially in a loop.

**Improvement:**
Use `joblib` or `ProcessPoolExecutor` to process chunks in parallel. Since the energy calculation is "embarrassingly parallel" (each block is independent), this scales linearly with CPU cores.

**Code Concept:**
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(processor.calculate_energy_field)(chunk) 
    for chunk in chunks
)
```

### 4. GPU Acceleration (CuPy)

**Current State:**
CPU only.

**Improvement:**
The entire pipeline relies on matrix arrays (`numpy`). Replacing `numpy` and `scipy.ndimage` with `cupy` and `cupyx.scipy.ndimage` allows the code to run on NVIDIA GPUs with almost zero code changes.
- **Energy Field:** GPU Gaussian blur is orders of magnitude faster.
- **Hessian:** 3x3 eigenvalue computation is instant on GPU.

**Implementation:**
Add a `use_gpu=True` flag to `SLAVVProcessor`.
```python
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi_gpu
    xp = cp
except ImportError:
    xp = np
    
# In code:
image_gpu = xp.asarray(image)
energy_gpu = ndi_gpu.gaussian_filter(image_gpu, sigma)
```

---

## Accuracy vs. Speed Trade-off

| Strategy | Speed Impact | Accuracy Impact | Notes |
| :--- | :--- | :--- | :--- |
| **Approximating PSF** | Slower (anisotropic filters) | **Higher** | Currently enabled. Disabling it (`approximating_PSF=False`) speeds up filters by 2x but ignores microscope physics. |
| **Grid Searching (scales)** | Slower (more scales) | **Higher** | Increasing `scales_per_octave` from 1.5 to 3.0 (as in tutorial) doubles compute time but improves vessel diameter estimation. |
| **Discrete vs. Continuous Tracing** | Slower (Continuous) | **Higher** | Continuous sub-voxel interpolation yields smoother vessel centerlines. Discrete is faster but "blocky". |

For the "Tutorial" quality, the parameters prioritize Accuracy (high scales, PSF modeling) over speed.
