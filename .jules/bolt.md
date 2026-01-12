# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-24 - [Optimizing Plotly 3D Traces]
**Learning:** Merging thousands of `go.Scatter3d` traces into a single trace using `None` separators reduces rendering time drastically (~4x speedup for 2000 edges).
**Action:** When implementing this optimization, note that per-edge opacity control is lost (merged traces support one `opacity` value). Update tests to check for merged data structures instead of individual trace properties.
