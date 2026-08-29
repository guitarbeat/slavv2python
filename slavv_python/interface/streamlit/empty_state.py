"""Shared empty-state copy for Streamlit pages that need a pipeline run."""

from __future__ import annotations

import streamlit as st

from slavv_python.interface.streamlit.navigation import switch_to
from slavv_python.schema import normalize_pipeline_result

MSG_NO_RUN = (
    "No processing results are available in this session. Open Processing and run "
    "an uploaded TIFF or a built-in sample."
)
MSG_NEED_EDGES = (
    "This page needs vertices and edges. In Processing, set Run through to "
    "Energy + Vertices + Edges (or the full network)."
)
MSG_NEED_NETWORK = (
    "This page needs a complete network. In Processing, set Run through to Full pipeline (Network)."
)


def require_processing_results() -> object | None:
    """Return session results, or warn and return None if the user has not run yet."""
    if "processing_results" not in st.session_state:
        st.warning(MSG_NO_RUN)
        if st.button(
            "Open Processing",
            icon=":material/arrow_forward:",
            key="empty_no_run_processing",
        ):
            switch_to("processing")
        return None
    return st.session_state["processing_results"]


def require_edges() -> object | None:
    """Return results that include vertices and edges, or warn and return None."""
    results = require_processing_results()
    if results is None:
        return None
    typed = normalize_pipeline_result(results)
    if typed.vertices is None or typed.edges is None:
        st.warning(MSG_NEED_EDGES)
        if st.button(
            "Complete through Edges",
            icon=":material/arrow_forward:",
            key="empty_need_edges_processing",
        ):
            switch_to("processing")
        return None
    return results


def require_network() -> object | None:
    """Return results that include a Network stage, or warn and return None."""
    results = require_processing_results()
    if results is None:
        return None
    typed = normalize_pipeline_result(results)
    if typed.vertices is None or typed.edges is None or typed.network is None:
        st.warning(MSG_NEED_NETWORK)
        if st.button(
            "Build the Network",
            icon=":material/arrow_forward:",
            key="empty_need_network_processing",
        ):
            switch_to("processing")
        return None
    return results
