## 2024-05-22 - Plotly 3D Trace Merging
**Learning:** Plotly `go.Scatter3d` performance degrades linearly with the number of traces. Merging disconnected lines into a single trace using `None` separators drastically reduces rendering time and DOM overhead.
**Action:** When visualizing networks or many segments, always check if they can be merged into a single trace with `None` separators, especially for 3D plots.
