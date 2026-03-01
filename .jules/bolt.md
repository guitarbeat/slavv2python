## 2024-05-23 - Wrapper Function Overhead
**Learning:** Wrapping a tight loop function like `compute_gradient_impl` with a wrapper `compute_gradient` that performs safety checks (casting, contiguity) on every call introduced a massive 22x overhead (660us vs 30us).
**Action:** Move data preparation/validation out of tight loops. Prepare data once in the caller and call the implementation directly.

## 2025-02-26 - Numpy Overhead on Scalars
**Learning:** Using `np.floor` and `astype(int)` on scalar values inside a tight loop (per-step tracing) introduces significant overhead compared to `math.floor` and `int()`. Inlining these checks reduced trace time by ~60%.
**Action:** Use Python built-ins for scalar arithmetic in tight loops; reserve NumPy for array operations.

## 2024-05-23 - [Pure Python Loop Optimization]
**Learning:** In tight loops (like gradient computation called 100k+ times), avoiding `np.clip` and intermediate array allocations in favor of manual scalar clamping and direct indexing yielded a ~5.8x speedup (32µs -> 5.6µs per call).
**Action:** When Numba or Cython isn't available/enabled, unroll small loops and use scalar math instead of numpy array operations for small fixed-size vectors (like 3D coordinates).
