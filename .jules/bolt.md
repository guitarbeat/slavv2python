## 2024-05-23 - Wrapper Function Overhead
**Learning:** Wrapping a tight loop function like `compute_gradient_impl` with a wrapper `compute_gradient` that performs safety checks (casting, contiguity) on every call introduced a massive 22x overhead (660us vs 30us).
**Action:** Move data preparation/validation out of tight loops. Prepare data once in the caller and call the implementation directly.

## 2024-05-24 - Plotly Custom Data Overhead
**Learning:** Passing a list of lists (e.g., `[[1, 2], [1, 2], ...]`) to Plotly's `customdata` attribute triggers extremely slow internal validation (likely type checking per item). Using a NumPy array `(N, M)` instead reduces overhead by ~60x.
**Action:** Always format `customdata`, `x`, `y`, `z`, and `color` inputs as NumPy arrays when working with large datasets in Plotly.
