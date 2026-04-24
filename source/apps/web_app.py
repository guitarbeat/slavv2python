from __future__ import annotations

import warnings

import streamlit as st

from .app_services import (
    _build_processing_run_dir,
    _has_full_network_results,
    _log_share_report_prepared_once,
    _render_run_dashboard,
    _update_run_task,
    cached_load_tiff_volume,
    generate_export_data,
    generate_share_report_data,
    load_run_snapshot,
)
from .app_services import (
    apply_curated_results as _apply_curated_results,
)
from .app_services import (
    run_interactive_curator as _run_interactive_curator,
)
from .web_app_analysis_page import show_analysis_page
from .web_app_curation_page import show_ml_curation_page
from .web_app_dashboard import (
    DASHBOARD_BREAKDOWN_SECTIONS,
    DASHBOARD_PLACEHOLDER,
    _dashboard_breakdown_frame,
    _dashboard_stage_frame,
)
from .web_app_dashboard_page import (
    DASHBOARD_ASSUMPTION,
    DASHBOARD_RELEASE_URL,
    DASHBOARD_REPO_URL,
    DashboardContext,
    _dashboard_context,
    _init_dashboard_state,
    _open_dashboard_metric_dialog,
    _render_dashboard_surface,
    _render_dashboard_surface_fragment,
    _toast_dashboard_feedback,
    show_dashboard_page,
)
from .web_app_processing_page import show_processing_page
from .web_app_shell import main
from .web_app_static_pages import show_about_page, show_home_page
from .web_app_visualization_page import (
    EXPORT_BUTTON_SPECS,
    show_visualization_page,
)
from .web_app_visualization_page import (
    _render_export_download as _render_export_download_impl,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SLAVV - Vascular Vectorization",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .parameter-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .section-header {
            font-size: 1.25rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
"""
)


def _render_export_download(
    column,
    *,
    run_dir: str | None,
    vertices,
    edges,
    network,
    parameters,
    export_spec: dict[str, str],
):
    """Facade wrapper that preserves monkeypatch-friendly app helpers."""
    return _render_export_download_impl(
        column,
        run_dir=run_dir,
        vertices=vertices,
        edges=edges,
        network=network,
        parameters=parameters,
        export_spec=export_spec,
        generate_export_data_fn=generate_export_data,
        update_run_task_fn=_update_run_task,
    )


__all__ = [
    "DASHBOARD_ASSUMPTION",
    "DASHBOARD_BREAKDOWN_SECTIONS",
    "DASHBOARD_PLACEHOLDER",
    "DASHBOARD_RELEASE_URL",
    "DASHBOARD_REPO_URL",
    "EXPORT_BUTTON_SPECS",
    "DashboardContext",
    "_apply_curated_results",
    "_build_processing_run_dir",
    "_dashboard_breakdown_frame",
    "_dashboard_context",
    "_dashboard_stage_frame",
    "_has_full_network_results",
    "_init_dashboard_state",
    "_log_share_report_prepared_once",
    "_open_dashboard_metric_dialog",
    "_render_dashboard_surface",
    "_render_dashboard_surface_fragment",
    "_render_export_download",
    "_render_run_dashboard",
    "_run_interactive_curator",
    "_toast_dashboard_feedback",
    "_update_run_task",
    "cached_load_tiff_volume",
    "generate_export_data",
    "generate_share_report_data",
    "load_run_snapshot",
    "main",
    "show_about_page",
    "show_analysis_page",
    "show_dashboard_page",
    "show_home_page",
    "show_ml_curation_page",
    "show_processing_page",
    "show_visualization_page",
]
