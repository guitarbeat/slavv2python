"""Shared empty-state copy for Streamlit pages that need a pipeline run."""

from __future__ import annotations

import streamlit as st

from slavv_python.schema import normalize_pipeline_result

MSG_NO_RUN = (
    "No processing results in this session. Open Image Processing, upload a TIFF, "
    "and run the pipeline."
)
MSG_NEED_EDGES = (
    "This step needs vertices and edges. On Image Processing, set Pipeline Target "
    "to at least Energy + Vertices + Edges."
)
MSG_NEED_NETWORK = (
    "This step needs a complete Network. On Image Processing, set Pipeline Target "
    "to Full Pipeline (Network)."
)


def require_processing_results() -> object | None:
    """Return session results, or warn and return None if the user has not run yet."""
    if "processing_results" not in st.session_state:
        st.warning(MSG_NO_RUN)
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
        return None
    return results
