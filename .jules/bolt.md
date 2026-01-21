## 2024-05-23 - Avoid Force-Contiguity on Hot Paths
**Learning:** `np.ascontiguousarray` with `dtype` conversion is extremely expensive when called in a tight loop (e.g. inside a pixel-level tracing algorithm), as it forces a full array copy if the input type mismatches (e.g. `float32` vs `float64`).
**Action:** Before converting arrays in hot paths, check if the underlying implementation actually requires specific dtypes or layouts. If using Numba, ensure the Numba signature matches the input data to avoid implicit casting, or hoist the conversion out of the loop.
