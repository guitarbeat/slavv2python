# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-23 - [Plotly 3D Trace Optimization]
**Learning:** Creating thousands of individual `go.Scatter3d` traces kills browser performance (WebGL context limit/overhead). Merging them into a single trace with `None` separators reduces rendering time by orders of magnitude (e.g. 4s -> 0.7s).
**Action:** Always merge line segments into single traces for 3D network visualizations. Use `None` to break lines. Restore tooltips using `text` array aligned with coordinates. Accept that per-edge opacity is not supported in merged mode.
