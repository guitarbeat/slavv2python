## 2024-05-23 - Wrapper Function Overhead
**Learning:** Wrapping a tight loop function like `compute_gradient_impl` with a wrapper `compute_gradient` that performs safety checks (casting, contiguity) on every call introduced a massive 22x overhead (660us vs 30us).
**Action:** Move data preparation/validation out of tight loops. Prepare data once in the caller and call the implementation directly.

## 2025-02-28 - Vectorized Color Sampling
**Learning:** Plotly's `px.colors.sample_colorscale` supports vectorized input (arrays) and is significantly faster (~17x) than iterating over values in Python.
**Action:** When mapping values to colors, always pass the full array to `sample_colorscale` instead of using list comprehensions.
