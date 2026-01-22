## 2024-05-22 - Hoisting Array Preparation in Loops
**Learning:** Even when using pure Python fallbacks for Numba/Cython functions, wrapper functions that enforce types (like `np.ascontiguousarray` or `dtype=float64`) can cause massive performance overhead if called inside tight loops.
**Action:** Always hoist array preparation/checking out of loops, especially for large 3D volumes. Pass prepared arrays to inner functions.
