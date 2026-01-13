# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-24 - [Optimizing Plotly 3D Traces]
**Learning:** Merging thousands of `go.Scatter3d` traces into a single trace using `None` separators reduces rendering time drastically (~4x speedup for 2000 edges).
**Action:** When implementing this optimization, note that per-edge opacity control is lost (merged traces support one `opacity` value). Update tests to check for merged data structures instead of individual trace properties.

## 2024-05-26 - [Plotly 2D Trace Optimization]
**Learning:** `go.Scatter` (2D) does not support array-based line coloring like `go.Scatter3d` or markers. To optimize 2D plots with many colored segments, we must group segments by color bin (for continuous values) or discrete ID into separate traces, rather than using a single merged trace with a color array.
**Action:** Use binning strategies (e.g. `np.digitize`) to group thousands of edges into a manageable number of traces (e.g., 50-100), reducing rendering overhead significantly.
