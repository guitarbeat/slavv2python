## 2024-05-23 - [Pure Python Loop Optimization]
**Learning:** In tight loops (like gradient computation called 100k+ times), avoiding `np.clip` and intermediate array allocations in favor of manual scalar clamping and direct indexing yielded a ~5.8x speedup (32µs -> 5.6µs per call).
**Action:** When Numba or Cython isn't available/enabled, unroll small loops and use scalar math instead of numpy array operations for small fixed-size vectors (like 3D coordinates).
