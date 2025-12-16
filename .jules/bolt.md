# Bolt's Journal

## 2024-05-22 - [Streamlit Caching]
**Learning:** Streamlit apps often re-run heavy computations on every interaction. `st.cache_data` or `st.cache_resource` are critical for performance.
**Action:** Always check for expensive function calls in `app.py` or core logic and verify if they are cached.
