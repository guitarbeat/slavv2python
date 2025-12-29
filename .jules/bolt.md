# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-24 - [Plotly Trace Merging]
**Learning:** Adding thousands of individual `go.Scatter3d` traces to a figure is extremely slow due to serialization and WebGL context overhead.
**Action:** Merge lines into a single trace using `None` separators and array-based coloring/customdata. This reduced render time from 3.5s to 1.4s for 2000 edges.
