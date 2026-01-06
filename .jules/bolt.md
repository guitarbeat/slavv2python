## 2024-05-23 - Plotly 3D Trace Merging
**Learning:** Creating thousands of individual `go.Scatter3d` traces is extremely slow due to Python-side overhead and WebGL context switching. Merging them into a single trace with `None` separators is a massive optimization (17x speedup).
**Action:** Always check if multiple traces can be merged into one using `None` separators, especially for 3D plots or large datasets. For variable coloring, use `line.color` array with a colorscale instead of separate traces. Note that this limits per-edge opacity support.
