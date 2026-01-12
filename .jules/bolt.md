# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-24 - [Plotly 3D Performance]
**Learning:** Creating thousands of individual `go.Scatter3d` traces kills performance (rendering & DOM size). Merging them into a single trace with `None` separators provides ~3x speedup.
**Action:** Always check for loops creating traces in visualization code. Merge traces where properties (like opacity) allow.
