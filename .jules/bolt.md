## 2024-05-23 - Wrapper Function Overhead
**Learning:** Wrapping a tight loop function like `compute_gradient_impl` with a wrapper `compute_gradient` that performs safety checks (casting, contiguity) on every call introduced a massive 22x overhead (660us vs 30us).
**Action:** Move data preparation/validation out of tight loops. Prepare data once in the caller and call the implementation directly.

## 2025-02-26 - Numpy Overhead on Scalars
**Learning:** Using `np.floor` and `astype(int)` on scalar values inside a tight loop (per-step tracing) introduces significant overhead compared to `math.floor` and `int()`. Inlining these checks reduced trace time by ~60%.
**Action:** Use Python built-ins for scalar arithmetic in tight loops; reserve NumPy for array operations.

## 2025-02-27 - Analytical vs Iterative Eigenvalues
**Learning:** `np.linalg.eigvalsh` is significantly slower (10x-50x) than an analytical solver for small matrices (3x3) when processed in Python loops, due to overhead and iterative nature. A vectorized analytical solution (using trigonometry) is vastly superior for this specific case.
**Action:** Prefer analytical solutions for small fixed-size matrix problems (2x2, 3x3) inside critical paths, but be careful with numerical stability (e.g. multiple roots).
