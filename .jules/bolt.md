# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-22 - [Plotly Performance]
**Learning:** Plotly trace overhead is massive for large networks. `go.Scatter3d` supports array-based line colors (allowing single-trace merge), but `go.Scattergl` in 2D does NOT (requires grouping by color).
**Action:** Always merge line segments into single traces with `None` separators when visualizing large networks (>100 edges). Use array coloring for 3D, group-by-color for 2D.
