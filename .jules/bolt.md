## 2024-05-23 - [Numpy Array Overhead in Hot Paths]
**Learning:** In tight loops like gradient calculation, creating small intermediate numpy arrays (e.g., via `copy()` or `np.clip`) and tuple indexing incurs massive overhead compared to scalar integer arithmetic and direct indexing.
**Action:** Unroll loops and use scalar indices for stencil operations in pure Python fallbacks.
