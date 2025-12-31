
## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.

## 2024-05-24 - [Visualization Caching]
**Learning:** Complex Plotly visualizations (specifically `NetworkVisualizer` methods like `plot_2d_network` and `plot_3d_network`) can take significant time (8+ seconds for 5k edges) to generate. Instantiating visualizers and regenerating figures on every Streamlit rerun degrades user experience.
**Action:** Wrap expensive visualization methods in `@st.cache_data` decorated functions in `app.py`. Ensure arguments are hashable (Streamlit handles numpy arrays well).
