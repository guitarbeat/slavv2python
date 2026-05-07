"""
Telemetry Dashboard Snippet

A simple Streamlit application to visualize normalized telemetry tables.
Run with: streamlit run workspace/scripts/analysis/dashboard_snippet.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SLAVV Telemetry Review", layout="wide")

st.title("📊 SLAVV Telemetry Review Dashboard")
st.markdown("---")

# Configuration
ANALYSIS_DIR = Path("03_Analysis")

if not ANALYSIS_DIR.exists():
    st.warning(f"Analysis directory `{ANALYSIS_DIR}` not found. Please run the normalizer first.")
    st.stop()

# Sidebar: File Selection
st.sidebar.header("Data Source")
files = list(ANALYSIS_DIR.glob("*.jsonl")) + list(ANALYSIS_DIR.glob("*.csv"))
if not files:
    st.sidebar.info("No normalized files found (.jsonl or .csv).")
    st.stop()

selected_file = st.sidebar.selectbox("Select Table", files, format_func=lambda x: x.name)


# Data Loading
@st.cache_data
def load_data(file_path):
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    return pd.read_csv(file_path)


df = load_data(selected_file)

# Dashboard Content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Data: {selected_file.name}")
    st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("Summary Metrics")
    st.write(f"**Total Records:** {len(df)}")

    # Specific parity metrics if available
    parity_fields = ["candidate_connection_count", "watershed_total_pairs"]
    available_parity = [f for f in parity_fields if f in df.columns]

    if available_parity:
        for field in available_parity:
            val = df[field].iloc[0] if not df.empty else 0
            st.metric(label=field.replace("_", " ").title(), value=int(val))
    else:
        st.info("No specific parity metrics found in this table.")

# Visualization
if not df.empty:
    st.markdown("---")
    st.subheader("Analysis Plots")

    # Try to plot something interesting if 'timestamp' or index exists
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        y_axis = st.selectbox("Select Metric to Plot", numeric_cols)
        st.line_chart(df[y_axis])
    else:
        st.info("No numeric columns available for plotting.")

st.sidebar.markdown("---")
st.sidebar.caption("SLAVV Parity Review Tool")
