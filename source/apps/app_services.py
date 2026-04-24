from __future__ import annotations

import streamlit as st
from source.io import load_tiff_volume
from source.runtime import load_run_snapshot

from .curation_services import apply_curated_results, run_interactive_curator
from .dashboard_services import render_run_dashboard
from .export_services import (
    build_run_task_dir,
    generate_export_data,
    generate_share_report_data,
    has_full_network_results,
    log_share_report_prepared_once,
    update_run_task,
)


@st.cache_data(show_spinner=False)
def cached_load_tiff_volume(file):
    """Cached wrapper for load_tiff_volume."""
    return load_tiff_volume(file)


_build_processing_run_dir = build_run_task_dir
_has_full_network_results = has_full_network_results
_log_share_report_prepared_once = log_share_report_prepared_once
_render_run_dashboard = render_run_dashboard
_update_run_task = update_run_task


__all__ = [
    "_build_processing_run_dir",
    "_has_full_network_results",
    "_log_share_report_prepared_once",
    "_render_run_dashboard",
    "_update_run_task",
    "apply_curated_results",
    "build_run_task_dir",
    "cached_load_tiff_volume",
    "generate_export_data",
    "generate_share_report_data",
    "has_full_network_results",
    "load_run_snapshot",
    "log_share_report_prepared_once",
    "render_run_dashboard",
    "run_interactive_curator",
    "update_run_task",
]


