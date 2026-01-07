# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-23 - [Plotly 3D Trace Optimization]
**Learning:** Creating thousands of individual `go.Scatter3d` traces is extremely slow and chokes the browser. Merging them into a single trace with `None` separators yields massive speedups (e.g., ~6.5x faster).
**Action:** Always optimize line plots with many segments by merging them into single traces, using `None` for separation and array-based coloring.
