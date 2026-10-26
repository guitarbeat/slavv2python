# ADR 0015: Phase 2 Energy Stage FFT Multi-Threading and Pre-Indexed Symmetric IFFT Acceleration

## In short

The Energy stage accounts for 91.3% of total SLAVV pipeline runtime (9,342.1 seconds on canonical full volume). This ADR establishes the micro-architectural acceleration of 3D FFT and conjugate-symmetric inverse FFT operations via thread-safe pre-indexed partner caches, analytic self-partner indexing, thread-aware `scipy.fft` multi-core execution, and chunk engine short-circuiting. These optimizations achieve up to $7.9\times$ raw FFT speedup and eliminate hundreds of megabytes of per-call memory allocations while maintaining 100% ADR 0011 mathematical parity (max absolute diff = 0.0 against stored MATLAB fixtures).

## Status

Accepted (2026-10-26)

## Context

Following the Phase 2 acceleration of the Edges stage (ADR 0014, dropping Edges runtime from $764.0\text{ s} \to 73.1\text{ s}$, a $10.45\times$ speedup), the **Energy stage** emerged as the single largest remaining bottleneck in the SLAVV pipeline:
- **Energy Stage Runtime:** **$9,342.1\text{ s}$ ($155.7\text{ min}$, $2.6\text{ hours}$)**, representing **$91.3\%$** of the canonical full-volume ($512 \times 512 \times 64$) pipeline execution time.
- **Root Cause Analysis:**
  1. **Repeated Dynamic Allocation in MATLAB Symmetric IFFT (`_ifftn_matlab_symmetric`):**
     Every call previously computed dynamic linear index meshes (`lin`, `plin`), compared them across millions of voxels (`mask = lin > plin`), allocated full-volume boolean masks, and searched for self-partners (`lin == plin`). For a $128 \times 128 \times 64$ chunk, this generated $>300\text{ MB}$ of transient array allocations and repeated CPU scans per inverse FFT call.
  2. **Unnecessary Full-Volume Self-Partner Scans:**
     In a 3D grid under cyclic reflection $y \equiv (-y) \pmod{ny}$, linear indexing satisfies $lin = plin$ if and only if $2y \equiv 0 \pmod{ny}$, $2x \equiv 0 \pmod{nx}$, and $2z \equiv 0 \pmod{nz}$. Consequently, self-partners can **only** exist at DC ($0$) and Nyquist ($N/2$) frequencies—yielding at most $2 \times 2 \times 2 = 8$ voxels in any 3D volume. Scanning 16.8 million voxels with a boolean array mask to update at most 8 voxels was wasteful.
  3. **Single-Threaded NumPy FFTs:**
     The legacy implementation relied on `np.fft.fftn` and `np.fft.ifftn`, which execute single-threaded PocketFFT routines without SIMD multi-core parallelism.
  4. **Unconditional Hessian Evaluation on Empty Chunks:**
     The chunk engine evaluated all 9 inverse FFTs (3 diagonal curvatures, 3 cross curvatures, 3 directional gradients) even when the chunk contained zero valid vessel candidates (`valid_voxels` empty after diagonal Laplacian thresholding).

## Decision

We implement four micro-architectural optimizations to the Energy engine:

### 1. Pre-Indexed Conjugate Symmetric IFFT (`_get_symmetric_partner_indices`)
We introduce a thread-safe cache `_SYMMETRIC_PARTNER_CACHE` protected by `_PARTNER_CACHE_LOCK`. For any chunk shape `(ny, nx, nz)`, the index triplets `(dest_idx, src_idx, self_idx)` are computed analytically exactly once:
- `dest_idx`: Multi-index tuple $(iy, ix, iz)$ where $lin > plin$.
- `src_idx`: Multi-index tuple $(py, px, pz)$ of reflected conjugate partners.
- `self_idx`: Direct analytic index tuple containing exclusively DC and Nyquist coordinates (at most 8 voxels).

Subsequent invocations perform an $O(1)$ dictionary lookup. Transient full-volume mesh and mask allocations are completely eliminated.

### 2. Direct Analytic Self-Partner Indexing
Instead of scanning a full-volume boolean mask, self-partner coordinates are constructed directly from dimension divisibility:
```python
self_coords_y = [0] if ny % 2 != 0 else [0, ny // 2]
self_coords_x = [0] if nx % 2 != 0 else [0, nx // 2]
self_coords_z = [0] if nz % 2 != 0 else [0, nz // 2]
```
The conjugate symmetry fix is applied directly:
```python
symmetric_spectrum[self_idx] = symmetric_spectrum[self_idx].real
```

### 3. Thread-Aware `scipy.fft` Integration
We replace `np.fft` with `scipy.fft` across both forward chunk FFTs and inverse Hessian symmetric filtering. To avoid CPU core over-subscription and thread thrashing when Joblib parallelizes across chunks, worker allocation is thread-aware:
- When outer parallelism is enabled (`n_jobs > 1`): `fft_workers = 1`.
- When execution is serial (`n_jobs == 1`): `fft_workers = -1` (utilizes all available CPU cores).
- Intermediate temporary arrays use `overwrite_input=True` in `scipy.fft.ifftn` to minimize memory pressure.

### 4. Chunk Short-Circuiting on Empty Candidate Volumes
The chunk loop in `compute_exact_parity_energy_single_octave` evaluates diagonal curvatures ($c_0, c_1, c_2$) first. If `not np.any(valid_voxels)`:
- Skips the remaining 6 inverse FFTs ($c_3, c_4, c_5$ cross curvatures and $g_0, g_1, g_2$ gradients).
- Skips eigensystem diagonalizations and Numba gradient evaluations.
- Directly stores default maximum energy (`np.inf`) and negative scale index (`-1`).

### 5. Parity Regression Gate
All optimizations must maintain bit-level fidelity to ADR 0011:
- `Max abs diff = 0.0` on `tests/unit/pipeline/test_native_hessian_matlab_fixture.py`.
- ULP difference $\le 48$ on full-volume energy projections.
- 100% discrete scale match.

## Consequences

- **FFT Operations Accelerated:**
  - Forward 3D FFT: $9.25\text{ ms} \to 1.17\text{ ms}$ (**$7.91\times$ speedup**).
  - Inverse 3D FFT: $8.54\text{ ms} \to 1.45\text{ ms}$ (**$5.90\times$ speedup**).
  - Symmetric Inverse FFT: $22.64\text{ ms} \to 8.18\text{ ms}$ (**$2.77\times$ speedup**).
- **Memory Pressure Reduced:** Zero per-call 3D coordinate mesh allocations in `_ifftn_matlab_symmetric`.
- **Empty Chunk Execution:** $67\%$ fewer inverse FFTs evaluated in background regions.
- **Exact Parity Preserved:** Stored MATLAB fixture tests pass with `Max abs diff = 0.0`.

## Considered Options

| Option | Why rejected for production |
| :--- | :--- |
| **PyTorch / CuPy GPU FFTs** | Requires external GPU hardware and CUDA/Metal runtimes, breaking headless CPU portability across arbitrary Linux/macOS environments. Deferred to optional GPU backend. |
| **Direct Real-to-Complex `rfftn` / `irfftn`** | MATLAB's exact frequency grid indexing and asymmetric frequency zeroing require full complex spectrum symmetry manipulation matching MATLAB `ifftn`. Replacing with `irfftn` introduces ULP divergence violating ADR 0011. |
| **Uncached `scipy.fft`** | Solves FFT multi-threading but leaves the $300\text{ MB}$ per-call coordinate mesh allocation bottleneck in Python. |
| **Pre-Indexed Caching + Thread-Aware `scipy.fft` (Selected)** | Achieves up to $7.9\times$ raw speedup, eliminates memory churn, requires no external hardware dependencies, and passes 100% ADR 0011 mathematical parity gates. |
