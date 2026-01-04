# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-23 - [Plotly 3D Trace Optimization]
**Learning:** Creating thousands of individual `go.Scatter3d` traces for edges (e.g. 5000 edges) creates massive overhead and can crash the browser. Merging them into a single trace with `None` separators reduces trace count to ~1 and renders almost instantly.
**Action:** When visualizing large networks in Plotly, always batch line segments into a single trace using `None` separators and array-based coloring.
