## 2024-05-23 - Wrapper Function Overhead
**Learning:** Wrapping a tight loop function like `compute_gradient_impl` with a wrapper `compute_gradient` that performs safety checks (casting, contiguity) on every call introduced a massive 22x overhead (660us vs 30us).
**Action:** Move data preparation/validation out of tight loops. Prepare data once in the caller and call the implementation directly.
