from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from slavv.analysis import AutomaticCurator, MLCurator
from slavv.apps.curation_state import (
    build_curation_stats_rows,
    summarize_processing_counts,
    sync_curated_processing_results,
)
from slavv.apps.share_report import compute_shareable_stats, record_share_event
from slavv.apps.web_app_artifacts import (
    _build_processing_run_dir,
    _has_full_network_results,
    _log_share_report_prepared_once,
    _update_run_task,
    generate_export_data,
    generate_share_report_data,
)
from slavv.apps.web_app_dashboard import (
    DASHBOARD_BREAKDOWN_SECTIONS,
    DASHBOARD_PLACEHOLDER,
    DASHBOARD_STAGE_ORDER,
    _build_dashboard_placeholder_trend,
    _dashboard_breakdown_frame,
    _dashboard_stage_frame,
    build_dashboard_backlog_frame,
    filter_dashboard_breakdown,
)
from slavv.core import SLAVVProcessor
from slavv.io import load_tiff_volume
from slavv.runtime import RunSnapshot, load_run_snapshot
from slavv.runtime.run_state import target_stage_progress
from slavv.utils import validate_parameters
from slavv.visualization import NetworkVisualizer

warnings.filterwarnings("ignore")
DASHBOARD_ASSUMPTION = (
    "Assumption: until dashboard metrics are specified, this view summarizes the active run, "
    "current network outputs, and share-report activity for the current session."
)
DASHBOARD_RELEASE_URL = "https://docs.streamlit.io/develop/quick-reference/release-notes"
DASHBOARD_REPO_URL = "https://github.com/UTFOIL/slavv2python"
EXPORT_BUTTON_SPECS = (
    {
        "format_type": "vmv",
        "label": "\U0001f4c4 Download VMV",
        "empty_label": "\U0001f4c4 Export VMV",
        "file_name": "network.vmv",
        "mime": "text/plain",
        "help": "Export network in VessMorphoVis (VMV) format",
        "artifact_key": "vmv_file",
    },
    {
        "format_type": "casx",
        "label": "\U0001f4c4 Download CASX",
        "empty_label": "\U0001f4c4 Export CASX",
        "file_name": "network.casx",
        "mime": "application/xml",
        "help": "Export network in CASX XML format",
        "artifact_key": "casx_file",
    },
    {
        "format_type": "csv",
        "label": "\U0001f4ca Download CSV (Zip)",
        "empty_label": "\U0001f4ca Export CSV",
        "file_name": "network_csv.zip",
        "mime": "application/zip",
        "help": "Export network data as Zipped CSVs (vertices & edges)",
        "artifact_key": "csv_archive",
    },
)


class DashboardContext(TypedDict):
    """Typed session-backed inputs for the dashboard surface."""

    run_dir: str | None
    snapshot: RunSnapshot | None
    results: dict[str, Any] | None
    share_metrics: dict[str, Any]
    dataset_name: str
    stats: dict[str, Any] | None


# Page configuration
st.set_page_config(
    page_title="SLAVV - Vascular Vectorization",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
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


@st.cache_data(show_spinner=False)
def cached_load_tiff_volume(file):
    """Cached wrapper for load_tiff_volume."""
    return load_tiff_volume(file)


def _render_export_download(
    column,
    *,
    run_dir: str | None,
    vertices,
    edges,
    network,
    parameters,
    export_spec: dict[str, str],
) -> None:
    """Render one export button using a shared table-driven config."""
    with column:
        if export_data := generate_export_data(
            vertices,
            edges,
            network,
            parameters,
            export_spec["format_type"],
        ):
            _update_run_task(
                run_dir,
                "exports",
                status="completed",
                detail="App export downloads prepared",
                artifacts={export_spec["artifact_key"]: export_spec["file_name"]},
            )
            st.download_button(
                label=export_spec["label"],
                data=export_data,
                file_name=export_spec["file_name"],
                mime=export_spec["mime"],
                help=export_spec["help"],
            )
        else:
            st.button(
                export_spec["empty_label"],
                disabled=True,
                help="Export generation failed",
            )


def _apply_curated_results(
    curated_vertices: dict[str, object],
    curated_edges: dict[str, object],
    *,
    curation_mode: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Sync curated vertices and edges into session state with a rebuilt network."""
    updated_results, baseline_counts, current_counts = sync_curated_processing_results(
        st.session_state["processing_results"],
        curated_vertices,
        curated_edges,
        baseline_counts=st.session_state.get("curation_baseline_counts"),
    )
    st.session_state["processing_results"] = updated_results
    st.session_state["curation_baseline_counts"] = baseline_counts
    st.session_state["last_curation_mode"] = curation_mode
    st.session_state.pop("share_report_prepared_signature", None)
    return baseline_counts, current_counts


def _render_run_dashboard(snapshot) -> None:
    """Render an in-app dashboard for the current run snapshot."""
    if snapshot is None:
        return

    current_stage = snapshot.current_stage or "idle"
    overall_pct = int(snapshot.overall_progress * 100)
    target_pct = int(target_stage_progress(snapshot) * 100)
    st.markdown("### Run Status")
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.metric("Run", snapshot.run_id)
    with col2:
        st.metric("Overall", f"{overall_pct}%")
    with col3:
        st.metric("Target", f"{target_pct}%")
    with col4:
        st.metric("Stage", current_stage)
    st.progress(overall_pct, text=f"Overall pipeline progress: {overall_pct}%")
    st.progress(
        target_pct,
        text=f"Progress to selected target ({snapshot.target_stage}): {target_pct}%",
    )

    stage_rows = []
    for stage_name in DASHBOARD_STAGE_ORDER:
        stage_snapshot = snapshot.stages.get(stage_name)
        if stage_snapshot is None:
            continue
        badge = "resumed" if stage_snapshot.resumed else "computed"
        detail = stage_snapshot.detail or stage_snapshot.substage or ""
        stage_rows.append(
            {
                "Stage": stage_name,
                "Status": stage_snapshot.status,
                "Progress": f"{int(stage_snapshot.progress * 100)}%",
                "Mode": badge,
                "Detail": detail,
            }
        )
    if stage_rows:
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)

    if snapshot.optional_tasks:
        st.markdown("### Optional Tasks")
        task_rows = [
            {
                "Task": name,
                "Status": task.status,
                "Progress": f"{int(task.progress * 100)}%",
                "Detail": task.detail,
            }
            for name, task in sorted(snapshot.optional_tasks.items())
        ]
        st.dataframe(pd.DataFrame(task_rows), use_container_width=True, hide_index=True)


def _init_dashboard_state() -> None:
    """Initialize local UI state for the dashboard shell."""
    st.session_state.setdefault("dashboard_focus", "Overview")
    st.session_state.setdefault("dashboard_sections", list(DASHBOARD_BREAKDOWN_SECTIONS))
    st.session_state.setdefault("dashboard_show_placeholders", True)
    st.session_state.setdefault("dashboard_auto_refresh", False)
    st.session_state.setdefault("dashboard_feedback", None)
    st.session_state.setdefault("dashboard_metric_requests", [])


def _dashboard_context() -> DashboardContext:
    """Load dashboard context from session state and run metadata."""
    run_dir = cast("str | None", st.session_state.get("current_run_dir"))
    snapshot = load_run_snapshot(run_dir) if run_dir else None
    results = cast("dict[str, Any] | None", st.session_state.get("processing_results"))
    share_metrics = cast("dict[str, Any]", st.session_state.get("share_report_metrics", {}))
    dataset_name = str(st.session_state.get("dataset_name", "No dataset loaded"))

    stats = None
    if results and _has_full_network_results(results):
        stats = compute_shareable_stats(
            results,
            image_shape=st.session_state.get("image_shape", (100, 100, 50)),
        )

    return {
        "run_dir": run_dir,
        "snapshot": snapshot,
        "results": results,
        "share_metrics": share_metrics,
        "dataset_name": dataset_name,
        "stats": stats,
    }


def _toast_dashboard_feedback() -> None:
    """Acknowledge dashboard feedback in the current session."""
    st.toast(
        "Thanks. The dashboard feedback was captured for this session.", icon=":material/thumb_up:"
    )


@st.dialog("Plan Dashboard Metrics")
def _open_dashboard_metric_dialog() -> None:
    """Collect a lightweight backlog request for dashboard follow-up."""
    with st.form("dashboard_metric_request", clear_on_submit=True):
        metric_name = st.text_input("Metric name", placeholder="Example: Export success rate")
        owner = st.selectbox(
            "Primary owner",
            ["Pipeline", "Operations", "Analysis", "Collaboration"],
            index=0,
        )
        priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1)
        notes = st.text_area(
            "Why it matters",
            placeholder="Describe the decision this metric should support.",
        )
        submitted = st.form_submit_button("Add backlog request")

    if not submitted:
        return
    if not metric_name.strip():
        st.warning("Add a metric name before submitting the request.")
        return

    requests = list(st.session_state.get("dashboard_metric_requests", []))
    requests.append(
        {
            "metric": metric_name.strip(),
            "owner": owner,
            "priority": priority,
            "notes": notes.strip(),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    st.session_state["dashboard_metric_requests"] = requests
    st.toast(f"Added '{metric_name.strip()}' to the dashboard backlog.", icon=":material/task_alt:")
    st.rerun()


def _render_dashboard_surface() -> None:
    """Render the core dashboard surface using the current session context."""
    context = _dashboard_context()
    run_dir = cast("str | None", context["run_dir"])
    snapshot = cast("RunSnapshot | None", context["snapshot"])
    results = cast("dict[str, Any] | None", context["results"])
    share_metrics = cast("dict[str, object]", context["share_metrics"])
    dataset_name = cast("str", context["dataset_name"])
    stats = cast("dict[str, Any] | None", context["stats"])

    source_mode = "Live run" if snapshot is not None else "Shell only"
    data_mode = "Network loaded" if stats is not None else "Awaiting metrics"

    badge_col, link_col = st.columns([3, 2], gap="large", vertical_alignment="center")
    with badge_col:
        st.badge("Streamlit 1.55")
        st.badge(source_mode)
        st.badge(data_mode)
        st.caption(
            f"Dataset: {dataset_name} | Sources: current run snapshot, session processing results, "
            "share-report counters"
        )
    with link_col:
        st.link_button("Release notes", DASHBOARD_RELEASE_URL, use_container_width=True)
        st.link_button("Repository", DASHBOARD_REPO_URL, use_container_width=True)

    st.space("small")

    if snapshot is None and stats is None:
        st.info(
            "No active run or completed network is loaded yet. Placeholder values stay visible so "
            "this layout can be extended once the dashboard scope is finalized."
        )

    overall_pct = int(snapshot.overall_progress * 100) if snapshot is not None else 0
    target_pct = int(target_stage_progress(snapshot) * 100) if snapshot is not None else 0
    strands_value = (
        DASHBOARD_PLACEHOLDER if stats is None else str(int(stats.get("num_strands", 0)))
    )
    total_length_value = (
        DASHBOARD_PLACEHOLDER
        if stats is None
        else f"{float(stats.get('total_length', 0.0)):.1f} um"
    )

    with st.container(border=True):
        st.subheader("Headline KPIs")
        col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
        with col1:
            st.metric("Run Progress", f"{overall_pct}%")
        with col2:
            st.metric("Target Progress", f"{target_pct}%")
        with col3:
            st.metric("Strands", strands_value)
        with col4:
            st.metric("Total Length", total_length_value)

    st.space("small")

    with st.container(border=True):
        st.subheader("Trends")
        trend_col1, trend_col2 = st.columns(2, gap="large")

        stage_frame = _dashboard_stage_frame(snapshot, run_dir=run_dir)
        stage_fig = px.line(stage_frame, x="Stage", y="Progress (%)", markers=True)
        stage_fig.update_traces(line={"width": 3}, marker={"size": 10})
        stage_fig.update_layout(
            height=320,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            xaxis_title="Pipeline stage",
            yaxis_title="Completion %",
            yaxis_range=[0, 100],
            showlegend=False,
        )
        with trend_col1:
            st.plotly_chart(stage_fig, use_container_width=True)
            st.caption("Pipeline progress is derived from the resumable run snapshot.")

        with trend_col2:
            if stats is not None and results is not None:
                try:
                    depth_fig = NetworkVisualizer().plot_depth_statistics(
                        results["vertices"],
                        results["edges"],
                        results.get("parameters", st.session_state.get("parameters", {})),
                    )
                    depth_fig.update_layout(
                        height=320,
                        margin={"l": 20, "r": 20, "t": 40, "b": 20},
                    )
                    st.plotly_chart(depth_fig, use_container_width=True)
                    st.caption("Depth statistics reuse the existing analysis visualization.")
                except Exception as exc:
                    st.plotly_chart(_build_dashboard_placeholder_trend(), use_container_width=True)
                    st.caption(
                        f"Depth trend placeholder shown because the live chart is unavailable: {exc}"
                    )
            else:
                st.plotly_chart(_build_dashboard_placeholder_trend(), use_container_width=True)
                st.caption(
                    "Placeholder trend slot for depth-resolved metrics once a complete network is available."
                )

    st.space("small")

    filtered_breakdown = filter_dashboard_breakdown(
        _dashboard_breakdown_frame(snapshot, stats, share_metrics, run_dir=run_dir),
        focus=st.session_state.get("dashboard_focus", "Overview"),
        selected_sections=st.session_state.get("dashboard_sections"),
        show_placeholders=st.session_state.get("dashboard_show_placeholders", True),
    )
    with st.container(border=True):
        st.subheader("Breakdown Table")
        if filtered_breakdown.empty:
            st.warning(
                "The current filters hide all rows. Re-enable placeholders or broaden the section filter."
            )
        else:
            st.dataframe(
                filtered_breakdown,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Section": st.column_config.TextColumn("Section", width="small"),
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Progress": st.column_config.ProgressColumn(
                        "Progress",
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                    ),
                    "Value": st.column_config.TextColumn("Value", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Source": st.column_config.TextColumn("Source", width="medium"),
                    "Notes": st.column_config.TextColumn("Notes", width="large"),
                },
            )

    source_tab, backlog_tab = st.tabs(["Source wiring", "Extension backlog"])
    with source_tab:
        st.caption("Dashboard controls are query-bound so the current view is easy to share.")
        st.json(dict(st.query_params))
        with st.expander("Assumptions and wiring points", icon=":material/lan:"):
            st.markdown(
                "\n".join(
                    [
                        "- The current shell focuses on the active run, network outputs, and share activity.",
                        "- `Pipeline` rows come from `run_snapshot.json`.",
                        "- `Network` rows come from `processing_results` via `compute_shareable_stats()`.",
                        "- Placeholder rows mark the next data sources to wire once entities are finalized.",
                    ]
                )
            )

    with backlog_tab:
        st.caption("Use this backlog to capture the next round of dashboard metrics and ownership.")
        st.data_editor(
            build_dashboard_backlog_frame(
                st.session_state.get("dashboard_metric_requests", []),
                repo_url=DASHBOARD_REPO_URL,
                release_url=DASHBOARD_RELEASE_URL,
            ),
            key="dashboard_backlog_editor",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Owner": st.column_config.SelectboxColumn(
                    "Owner",
                    options=["Pipeline", "Operations", "Analysis", "Collaboration"],
                    width="small",
                ),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["High", "Medium", "Low"],
                    width="small",
                ),
                "Tracked": st.column_config.CheckboxColumn("Tracked"),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["TODO", "Requested", "Ready"],
                    width="small",
                ),
                "Reference": st.column_config.LinkColumn("Reference", display_text="Open"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
            disabled=["Metric", "Reference"],
        )
        st.feedback("stars", key="dashboard_feedback", on_change=_toast_dashboard_feedback)


@st.fragment(run_every="20s")
def _render_dashboard_surface_fragment() -> None:
    """Refresh the dashboard surface independently when auto-refresh is enabled."""
    _render_dashboard_surface()


def show_dashboard_page():
    """Display an extendable dashboard shell for the current SLAVV session."""
    st.markdown('<h2 class="section-header">Operations Dashboard</h2>', unsafe_allow_html=True)
    st.info(DASHBOARD_ASSUMPTION)
    _init_dashboard_state()

    controls_col, action_col = st.columns([3, 1], gap="large", vertical_alignment="bottom")
    with controls_col, st.popover("Dashboard controls", use_container_width=True):
        st.segmented_control(
            "Focus",
            ["Overview", "Pipeline", "Network"],
            key="dashboard_focus",
            bind="query-params",
        )
        st.pills(
            "Sections",
            list(DASHBOARD_BREAKDOWN_SECTIONS),
            key="dashboard_sections",
            selection_mode="multi",
            bind="query-params",
        )
        st.toggle(
            "Show placeholder rows",
            key="dashboard_show_placeholders",
            bind="query-params",
        )
        st.toggle(
            "Auto-refresh dashboard",
            key="dashboard_auto_refresh",
            bind="query-params",
        )
        st.caption("These controls sync with the URL so the current view is easy to share.")
    with action_col:
        if st.button("Plan metrics", use_container_width=True):
            _open_dashboard_metric_dialog()

    if st.session_state.get("dashboard_auto_refresh", False):
        _render_dashboard_surface_fragment()
    else:
        _render_dashboard_surface()


def _run_interactive_curator(energy_data, vertices_data, edges_data, backend="qt"):
    """Import desktop curator backends lazily so the web app can load without GUI deps."""
    backend_name = str(backend).strip().lower()
    if backend_name in {"qt", "qt_pyvista", "pyvista"}:
        from slavv.visualization.interactive_curator import run_curator

        return run_curator(energy_data, vertices_data, edges_data)
    if backend_name == "napari":
        from slavv.visualization.napari_curator import run_curator_napari

        return run_curator_napari(energy_data, vertices_data, edges_data)
    raise ValueError("curator backend must be 'qt' or 'napari'")


def main():
    """Main application function"""

    # Header
    st.markdown(
        '<h1 class="main-header">🩸 SLAVV - Vascular Vectorization System</h1>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    **Segmentation-Less, Automated, Vascular Vectorization** - A comprehensive tool for analyzing
    vascular networks from grayscale, volumetric microscopy images.

    This Python/Streamlit implementation is based on the MATLAB SLAVV algorithm by Samuel Alexander Mihelic.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "⚙️ Image Processing",
            "🤖 ML Curation",
            "📊 Visualization",
            "📈 Analysis",
            "Info: About",
        ],
    )

    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "⚙️ Image Processing":
        show_processing_page()
    elif page == "🤖 ML Curation":
        show_ml_curation_page()
    elif page == "📊 Visualization":
        show_visualization_page()
    elif page == "📈 Analysis":
        show_analysis_page()
    elif page == "Info: About":
        show_about_page()


def show_home_page():
    """Display the home page with overview and quick start"""

    show_dashboard_page()
    st.divider()
    st.markdown('<h2 class="section-header">Welcome to SLAVV</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large", vertical_alignment="top")

    with col1:
        st.markdown("""
        ### 🔬 What is SLAVV?

        SLAVV (Segmentation-Less, Automated, Vascular Vectorization) is a sophisticated algorithm
        for extracting and analyzing vascular networks from 3D microscopy images. The method works
        through four main steps:

        1. **Energy Image Formation** - Multi-scale Hessian-based filtering to enhance vessel centerlines
        2. **Vertex Extraction** - Detection of vessel bifurcations and endpoints as local energy minima
        3. **Edge Extraction** - Tracing vessel segments between vertices through gradient descent
        4. **Network Construction** - Assembly of edges into connected vascular strands

        ### 🚀 Key Features

        - **Multi-scale Analysis** - Detects vessels across a wide range of sizes
        - **PSF Correction** - Accounts for microscope point spread function
        - **ML Curation** - Machine learning-assisted quality control
        - **Comprehensive Statistics** - Detailed network analysis and metrics
        - **Multiple Export Formats** - VMV, CASX, and custom formats
        - **Interactive Visualization** - 2D and 3D network rendering
        """)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        **✅ Ready to get started?**

        1. Navigate to **Image Processing** to upload and process your TIFF images
        2. Use **ML Curation** to refine vertex and edge detection
        3. Explore results in **Visualization** and **Analysis** pages
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2, st.container(height=400):
        st.markdown("### 📊 Quick Stats")

        # Sample statistics (would be replaced with actual data)
        st.metric("Supported Image Types", "TIFF", help="3D grayscale TIFF images")
        st.metric("Processing Steps", "4", help="Energy → Vertices → Edges → Network")
        st.metric("Export Formats", "5+", help="VMV, CASX, MAT, CSV, JSON")

        st.markdown("### 🔧 System Requirements")
        st.markdown("""
            - **Input**: 3D TIFF images
            - **Memory**: Depends on image size
            - **Processing**: Multi-threaded CPU
            - **Output**: Vector networks + statistics
            """)

        st.markdown("### 📚 Documentation")
        st.markdown("""
            - [Algorithm Overview](#)
            - [Parameter Guide](#)
            - [Export Formats](#)
            - [Troubleshooting](#)
            """)

        st.markdown("### 🎯 Workflow Control")
        st.markdown("""
            Like the original MATLAB scripts (`StartWorkflow`/`FinalWorkflow`), you can
            pause the pipeline early to inspect intermediate results or force the pipeline
            to recalculate specific steps to test parameter changes.
            """)


def show_processing_page():
    """Display the image processing page"""

    st.markdown('<h2 class="section-header">Image Processing Pipeline</h2>', unsafe_allow_html=True)

    # File upload
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a TIFF file",
        type=["tif", "tiff"],
        help="Upload a 3D grayscale TIFF image of vascular structures",
    )

    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type,
        }
        st.json(file_details)

    # Processing parameters
    st.markdown('<h3 class="section-header">Processing Parameters</h3>', unsafe_allow_html=True)
    with st.popover("Parameter tips", width=300):
        st.write(
            "Use the tabs below to adjust microscopy, vessel size, processing, "
            "and advanced options. Defaults are provided for typical datasets."
        )

    # Create tabs for parameter categories
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔬 Microscopy", "📏 Vessel Sizes", "⚙️ Processing", "🔬 Advanced"]
    )

    with tab1:
        st.markdown("#### Microscopy Parameters")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            microns_per_voxel_y = st.number_input(
                "Y voxel size (μm)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Physical size of one voxel in Y dimension. (MATLAB: microns_per_voxel(1))",
            )
            microns_per_voxel_x = st.number_input(
                "X voxel size (μm)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Physical size of one voxel in X dimension. (MATLAB: microns_per_voxel(2))",
            )
            microns_per_voxel_z = st.number_input(
                "Z voxel size (μm)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Physical size of one voxel in Z dimension. (MATLAB: microns_per_voxel(3))",
            )

        with col2:
            approximating_PSF = st.checkbox(
                "Approximate PSF",
                value=True,
                help="Account for microscope point spread function using Zipfel et al. model. (MATLAB: approximating_PSF)",
            )

            if approximating_PSF:
                numerical_aperture = st.number_input(
                    "Numerical Aperture",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.95,
                    step=0.01,
                    help="Numerical aperture of the microscope objective. (MATLAB: numerical_aperture)",
                )

                excitation_wavelength = st.number_input(
                    "Excitation wavelength (μm)",
                    min_value=0.4,
                    max_value=3.0,
                    value=1.3,
                    step=0.1,
                    help="Laser excitation wavelength. Typical range: 0.7-3.0 μm for two-photon microscopy. (MATLAB: excitation_wavelength_in_microns)",
                )

                # Warning for wavelength outside typical range
                if not (0.7 <= excitation_wavelength <= 3.0):
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning(
                        "⚠️ Excitation wavelength outside typical range (0.7-3.0 μm). This range is typical for two-photon microscopy. Please verify this value."
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                sample_index_of_refraction = st.number_input(
                    "Sample refractive index",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.33,
                    step=0.01,
                    help="Refractive index of the sample medium (e.g., 1.33 for water). (MATLAB: sample_index_of_refraction)",
                )

    with tab2:
        st.markdown("#### Vessel Size Parameters")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            radius_smallest = st.number_input(
                "Smallest vessel radius (μm)",
                min_value=0.1,
                max_value=100.0,
                value=1.5,
                step=0.1,
                help="Radius of the smallest vessel to be detected in microns. (MATLAB: radius_of_smallest_vessel_in_microns)",
            )

            radius_largest = st.number_input(
                "Largest vessel radius (μm)",
                min_value=1.0,
                max_value=500.0,
                value=50.0,
                step=1.0,
                help="Radius of the largest vessel to be detected in microns. (MATLAB: radius_of_largest_vessel_in_microns)",
            )

            if radius_largest <= radius_smallest:
                st.error("❌ Largest radius must be greater than smallest radius")

        with col2:
            scales_per_octave = st.number_input(
                "Scales per octave",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Number of vessel sizes to detect per doubling of the radius cubed. (MATLAB: scales_per_octave)",
            )

            # Calculate and display scale information
            if radius_largest > radius_smallest:
                volume_ratio = (radius_largest / radius_smallest) ** 3
                n_scales = int(np.log(volume_ratio) / np.log(2) * scales_per_octave) + 3
                st.info(f"📊 This will generate approximately {n_scales} scales")

    with tab3:
        st.markdown("#### Processing Parameters")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            energy_upper_bound = st.number_input(
                "Energy upper bound",
                min_value=-10.0,
                max_value=0.0,
                value=0.0,
                step=0.1,
                help="Maximum energy value for vertex detection (negative values). (MATLAB: energy_upper_bound)",
            )

            space_strel_apothem = st.number_input(
                "Spatial structuring element",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Minimum spacing between detected vertices (in voxels). (MATLAB: space_strel_apothem)",
            )

            length_dilation_ratio = st.number_input(
                "Length dilation ratio",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Ratio of rendering length to detection length for volume exclusion. (MATLAB: length_dilation_ratio)",
            )

        with col2:
            number_of_edges_per_vertex = st.number_input(
                "Edges per vertex",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Maximum number of edge traces per seed vertex. (MATLAB: number_of_edges_per_vertex)",
            )

            max_voxels_per_node = st.number_input(
                "Max voxels per node",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=1000,
                help="Maximum voxels per computational node for parallel processing. (MATLAB: max_voxels_per_node_energy)",
            )

    with tab4:
        st.markdown("#### Advanced Parameters")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            gaussian_to_ideal_ratio = st.slider(
                "Gaussian to ideal ratio",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Standard deviation of the Gaussian kernel per the total object length for objects that are much larger than the PSF. (MATLAB: gaussian_to_ideal_ratio)",
            )

            spherical_to_annular_ratio = st.slider(
                "Spherical to annular ratio",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Weighting factor of the spherical pulse over the combined weights of spherical and annular pulses. (MATLAB: spherical_to_annular_ratio)",
            )

        with col2:
            step_size_per_origin_radius = st.number_input(
                "Step size ratio",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Edge tracing step size relative to origin vertex radius. (MATLAB: step_size_per_origin_radius)",
            )

            max_edge_energy = st.number_input(
                "Max edge energy",
                min_value=-10.0,
                max_value=0.0,
                value=0.0,
                step=0.1,
                help="Maximum energy threshold for edge tracing. (MATLAB: max_edge_energy)",
            )

    # Processing button and results
    st.markdown('<h3 class="section-header">Processing</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        stop_after_options = {
            "Energy Field Only": "energy",
            "Energy + Vertices": "vertices",
            "Energy + Vertices + Edges": "edges",
            "Full Pipeline (Network)": "network",
        }
        stop_after_selection = st.selectbox(
            "Pipeline Target",
            options=list(stop_after_options.keys()),
            index=3,
            help="Stop the pipeline early after completing this stage. Useful for tweaking parameters.",
        )
        stop_after_val = stop_after_options[stop_after_selection]

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_rerun_stage = st.selectbox(
            "Force Recalculation From:",
            options=["None", "energy", "vertices", "edges", "network"],
            index=0,
            help="Ignore cached results and recalculate from this stage onwards. Leave as 'None' to use cached files if available.",
        )

    current_snapshot = (
        load_run_snapshot(st.session_state.get("current_run_dir"))
        if st.session_state.get("current_run_dir")
        else None
    )
    if current_snapshot is not None:
        _render_run_dashboard(current_snapshot)

    if uploaded_file is not None:
        if st.button("🚀 Start Processing", type="primary", width=250):
            # Collect parameters
            parameters = {
                "microns_per_voxel": [
                    microns_per_voxel_y,
                    microns_per_voxel_x,
                    microns_per_voxel_z,
                ],
                "radius_of_smallest_vessel_in_microns": radius_smallest,
                "radius_of_largest_vessel_in_microns": radius_largest,
                "approximating_PSF": approximating_PSF,
                "scales_per_octave": scales_per_octave,
                "energy_upper_bound": energy_upper_bound,
                "space_strel_apothem": space_strel_apothem,
                "length_dilation_ratio": length_dilation_ratio,
                "number_of_edges_per_vertex": number_of_edges_per_vertex,
                "max_voxels_per_node_energy": max_voxels_per_node,
                "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
                "spherical_to_annular_ratio": spherical_to_annular_ratio,
                "step_size_per_origin_radius": step_size_per_origin_radius,
                "max_edge_energy": max_edge_energy,
            }

            if approximating_PSF:
                parameters.update(
                    {
                        "numerical_aperture": numerical_aperture,
                        "excitation_wavelength_in_microns": excitation_wavelength,
                        "sample_index_of_refraction": sample_index_of_refraction,
                    }
                )

            # Validate parameters and process image
            try:
                validated_params = validate_parameters(parameters)
                st.success("✅ Parameters validated successfully")

                with st.status("Processing image...", expanded=True) as status:
                    status.update(label="Loading image...", state="running")
                    try:
                        image = cached_load_tiff_volume(uploaded_file)
                        st.success(f"✅ Image loaded successfully with shape: {image.shape}")
                    except ValueError as e:
                        st.error(f"❌ Error loading TIFF file: {e}")
                        st.stop()

                    processor = SLAVVProcessor()

                    dashboard_placeholder = st.empty()
                    run_dir = _build_processing_run_dir(
                        uploaded_file.getvalue(),
                        validated_params,
                    )

                    def event_cb(event) -> None:
                        state = "complete" if event.status.startswith("completed") else "running"
                        label = event.detail or f"{event.stage} {int(event.stage_progress * 100)}%"
                        status.update(label=label, state=state)
                        with dashboard_placeholder.container():
                            _render_run_dashboard(event.snapshot)

                    results = processor.process_image(
                        image,
                        validated_params,
                        event_callback=event_cb,
                        run_dir=run_dir,
                        stop_after=stop_after_val,
                        force_rerun_from=force_rerun_stage if force_rerun_stage != "None" else None,
                    )

                    final_snapshot = load_run_snapshot(run_dir) if run_dir else None
                    with dashboard_placeholder.container():
                        _render_run_dashboard(final_snapshot)
                    status.update(
                        label=f"Processing finished at target: {stop_after_val}",
                        state="complete",
                    )

                # Store results in session state
                st.session_state["processing_results"] = results
                st.session_state["parameters"] = validated_params
                st.session_state["image_shape"] = image.shape
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["current_run_dir"] = run_dir
                st.session_state["run_snapshot"] = (
                    final_snapshot.to_dict() if final_snapshot is not None else None
                )
                st.session_state.pop("curation_baseline_counts", None)
                st.session_state.pop("last_curation_mode", None)
                st.session_state.pop("share_report_prepared_signature", None)
                _render_run_dashboard(final_snapshot)

                # Cleanup state if we stopped early so UI doesn't crash trying to render old networks
                if stop_after_val != "network":
                    st.warning(
                        f"⚠️ Pipeline halted early at '{stop_after_val}'. Downstream results (if any) are not available."
                    )

                # Display results summary
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("🎉 Processing stage completed successfully!")
                st.markdown("</div>", unsafe_allow_html=True)

                # Results summary
                col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")

                with col1:
                    vertices_count = len(results.get("vertices", {}).get("positions", []))
                    st.metric(
                        "Vertices Found",
                        vertices_count if "vertices" in results else "N/A",
                        help="Total vertices detected in the volume",
                    )

                with col2:
                    edges_count = len(results.get("edges", {}).get("traces", []))
                    st.metric(
                        "Edges Extracted",
                        edges_count if "edges" in results else "N/A",
                        help="Number of vessel segments traced",
                    )

                with col3:
                    strands_count = len(results.get("network", {}).get("strands", []))
                    st.metric(
                        "Network Strands",
                        strands_count if "network" in results else "N/A",
                        help="Connected components in the network",
                    )

                with col4:
                    bif_count = len(results.get("network", {}).get("bifurcations", []))
                    st.metric(
                        "Bifurcations",
                        bif_count if "network" in results else "N/A",
                        help="Detected branching points",
                    )

            except Exception as e:
                st.error(f"❌ Processing failed: {e!s}")

    else:
        st.info("👆 Please upload a TIFF file to begin processing")


def show_ml_curation_page():
    """Display the ML curation page"""

    st.markdown('<h2 class="section-header">Machine Learning Curation</h2>', unsafe_allow_html=True)

    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return

    results = st.session_state["processing_results"]
    if "vertices" not in results or "edges" not in results:
        st.warning(
            "⚠️ Curation requires both vertices and edges to be extracted. Please run the pipeline at least up to the 'edges' stage."
        )
        return
    st.markdown("""
    Use machine learning algorithms or heuristic rules to automatically curate and refine the detected vertices and edges.
    This step helps improve the accuracy of the vectorization by removing false positives and enhancing
    true vascular structures. This functionality is based on `MLDeployment.py` and `MLLibrary.py` from the original MATLAB repository.
    """)

    results = st.session_state["processing_results"]
    st.session_state["parameters"]

    st.markdown("### 🎯 Curation Options")
    curation_type = st.radio(
        "Select Curation Type:",
        ("Interactive (Manual GUI)", "Automatic (Rule-based)", "Machine Learning (Model-based)"),
        help="Choose how to curate nodes/edges. Interactive opens a 3D pop-up window.",
    )

    if curation_type == "Interactive (Manual GUI)":
        st.markdown("#### Interactive 3D Curation")
        st.info(
            "Launch the 3D Graphical Curator Interface to manually add or delete vertices and edges."
        )

        curator_backend_label = st.selectbox(
            "Interactive curator backend",
            ("Qt/PyVista (default)", "napari (experimental)"),
            help=(
                "Qt/PyVista preserves the current curator. napari is an experimental "
                "prototype with simpler image, point, and path editing."
            ),
        )
        curator_backend = "napari" if curator_backend_label.startswith("napari") else "qt"

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Launch Interactive Curator", type="primary", width=250):
                _update_run_task(
                    st.session_state.get("current_run_dir"),
                    "manual_curation",
                    status="running",
                    detail="Interactive curator launched",
                )
                # Put up a status so user knows to look for the window
                with st.status(
                    "Interactive Curator running in new window...", expanded=True
                ) as status:
                    st.warning(
                        "⚠️ Please check your taskbar for the new 3D window. Closing the window will save and continue."
                    )

                    # Launch the blocking desktop curator window.
                    curated_vertices, curated_edges = _run_interactive_curator(
                        results["energy_data"],
                        results["vertices"],
                        results["edges"],
                        backend=curator_backend,
                    )

                    status.update(label="Rebuilding network after curation...", state="running")
                    try:
                        baseline_counts, current_counts = _apply_curated_results(
                            curated_vertices,
                            curated_edges,
                            curation_mode="Interactive (Manual GUI)",
                        )
                    except Exception as exc:
                        _update_run_task(
                            st.session_state.get("current_run_dir"),
                            "manual_curation",
                            status="failed",
                            detail=f"Interactive curation could not rebuild the network: {exc!s}",
                        )
                        st.error(
                            "Curated vertices and edges were not applied because the network "
                            f"could not be rebuilt: {exc!s}"
                        )
                        st.stop()
                    _update_run_task(
                        st.session_state.get("current_run_dir"),
                        "manual_curation",
                        status="completed",
                        detail="Interactive curation saved and network rebuilt",
                    )

                    status.update(label="Interactive Curation complete!", state="complete")
                    st.success("✅ Interactive edits saved!")

                    st.caption(
                        "The downstream network, exports, and share report now use the curated "
                        "vertices and edges."
                    )

                    # Rerender metrics
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        st.metric(
                            "Vertices",
                            current_counts["Vertices"],
                            delta=current_counts["Vertices"] - baseline_counts["Vertices"],
                        )
                    with c2:
                        st.metric(
                            "Edges",
                            current_counts["Edges"],
                            delta=current_counts["Edges"] - baseline_counts["Edges"],
                        )

    elif curation_type == "Automatic (Rule-based)":
        st.markdown("#### Automatic Curation Parameters")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            vertex_energy_threshold = st.number_input(
                "Vertex Energy Threshold",
                min_value=-10.0,
                max_value=0.0,
                value=-0.1,
                step=0.01,
                help="Vertices with energy above this threshold will be removed.",
            )
            min_vertex_radius = st.number_input(
                "Minimum Vertex Radius (μm)",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                step=0.1,
                help="Vertices with radius below this will be removed.",
            )
        with col2:
            boundary_margin = st.number_input(
                "Boundary Margin (voxels)",
                min_value=0,
                max_value=20,
                value=5,
                step=1,
                help="Vertices too close to image boundaries will be removed.",
            )
            contrast_threshold = st.number_input(
                "Local Contrast Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Vertices in low-contrast regions will be removed.",
            )
            min_edge_length = st.number_input(
                "Minimum Edge Length (μm)",
                min_value=0.1,
                max_value=20.0,
                value=2.0,
                step=0.1,
                help="Edges shorter than this will be removed.",
            )
            max_edge_tortuosity = st.number_input(
                "Maximum Edge Tortuosity",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Edges with tortuosity above this will be removed.",
            )
            max_connection_distance = st.number_input(
                "Max Connection Distance (μm)",
                min_value=0.1,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="Edges not properly connected to vertices within this distance will be removed.",
            )

        auto_curation_params = {
            "vertex_energy_threshold": vertex_energy_threshold,
            "min_vertex_radius": min_vertex_radius,
            "boundary_margin": boundary_margin,
            "contrast_threshold": contrast_threshold,
            "min_edge_length": min_edge_length,
            "max_edge_tortuosity": max_edge_tortuosity,
            "max_connection_distance": max_connection_distance,
            "image_shape": st.session_state["image_shape"],  # Pass image shape for boundary check
        }

        if st.button("🚀 Start Automatic Curation", type="primary", width=250):
            _update_run_task(
                st.session_state.get("current_run_dir"),
                "automatic_curation",
                status="running",
                detail="Automatic curation started",
            )
            with st.status(
                "Performing automatic curation...",
                expanded=True,
            ) as status:
                curator = AutomaticCurator()

                curated_vertices = curator.curate_vertices_automatic(
                    results["vertices"], results["energy_data"], auto_curation_params
                )

                curated_edges = curator.curate_edges_automatic(
                    results["edges"], curated_vertices, auto_curation_params
                )

                status.update(label="Rebuilding network after curation...", state="running")
                try:
                    baseline_counts, current_counts = _apply_curated_results(
                        curated_vertices,
                        curated_edges,
                        curation_mode="Automatic (Rule-based)",
                    )
                except Exception as exc:
                    _update_run_task(
                        st.session_state.get("current_run_dir"),
                        "automatic_curation",
                        status="failed",
                        detail=f"Automatic curation could not rebuild the network: {exc!s}",
                    )
                    st.error(
                        "Curated vertices and edges were not applied because the network "
                        f"could not be rebuilt: {exc!s}"
                    )
                    st.stop()
                _update_run_task(
                    st.session_state.get("current_run_dir"),
                    "automatic_curation",
                    status="completed",
                    detail="Automatic curation complete and network rebuilt",
                )

                st.success("✅ Automatic curation complete!")
                status.update(label="Automatic curation complete!", state="complete")
                st.caption(
                    "The downstream network, exports, and share report now use the curated "
                    "vertices and edges."
                )

                col1, col2 = st.columns(2, gap="small")
                with col1:
                    st.metric(
                        "Vertices",
                        current_counts["Vertices"],
                        delta=current_counts["Vertices"] - baseline_counts["Vertices"],
                        help="Change relative to the pre-curation baseline",
                    )
                with col2:
                    st.metric(
                        "Edges",
                        current_counts["Edges"],
                        delta=current_counts["Edges"] - baseline_counts["Edges"],
                        help="Change relative to the pre-curation baseline",
                    )

    elif curation_type == "Machine Learning (Model-based)":
        st.markdown("#### Machine Learning Curation Parameters")
        st.info("Upload pre-trained models or provide CSV training data to train new classifiers.")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.selectbox(
                "Vertex curation method",
                ["machine-auto"],  # Only machine-auto for now
                help="Choose how to curate detected vertices. Corresponds to `VertexCuration` parameter in MATLAB.",
            )

            vertex_model_file = st.file_uploader(
                "Vertex model (.joblib)",
                type=["joblib", "pkl"],
                help="Upload a pre-trained vertex classifier",
            )

            vertex_training_data = st.file_uploader(
                "Vertex training data (.csv)",
                type=["csv"],
                help="CSV with vertex features and a 'label' column",
            )

            vertex_confidence_threshold = st.slider(
                "Vertex Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for keeping vertices",
            )

        with col2:
            st.selectbox(
                "Edge curation method",
                ["machine-auto"],  # Only machine-auto for now
                help="Choose how to curate detected edges. Corresponds to `EdgeCuration` parameter in MATLAB.",
            )

            edge_model_file = st.file_uploader(
                "Edge model (.joblib)",
                type=["joblib", "pkl"],
                help="Upload a pre-trained edge classifier",
            )

            edge_training_data = st.file_uploader(
                "Edge training data (.csv)",
                type=["csv"],
                help="CSV with edge features and a 'label' column",
            )

            edge_confidence_threshold = st.slider(
                "Edge Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for keeping edges",
            )

        if st.button("📚 Train Models", type="secondary", width=250):
            if vertex_training_data is None and edge_training_data is None:
                st.error("Please upload training data for vertices, edges, or both.")
            else:
                _update_run_task(
                    st.session_state.get("current_run_dir"),
                    "ml_training",
                    status="running",
                    detail="Training ML models",
                )
                with st.status("Training ML models...", expanded=True) as status:
                    ml_curator = MLCurator()
                    if vertex_training_data is not None:
                        df_v = pd.read_csv(vertex_training_data)
                        X_v = df_v.drop(columns=["label"]).values
                        y_v = df_v["label"].values
                        res_v = ml_curator.train_vertex_classifier(X_v, y_v)
                        st.write(f"Vertex test accuracy: {res_v['test_accuracy']:.3f}")
                    if edge_training_data is not None:
                        df_e = pd.read_csv(edge_training_data)
                        X_e = df_e.drop(columns=["label"]).values
                        y_e = df_e["label"].values
                        res_e = ml_curator.train_edge_classifier(X_e, y_e)
                        st.write(f"Edge test accuracy: {res_e['test_accuracy']:.3f}")
                    st.session_state["ml_curator"] = ml_curator
                    _update_run_task(
                        st.session_state.get("current_run_dir"),
                        "ml_training",
                        status="completed",
                        detail="ML models trained",
                    )
                    status.update(label="Training complete!", state="complete")
                    st.success("✅ Models trained!")

        if st.button("🤖 Start ML Curation", type="primary", width=250):
            _update_run_task(
                st.session_state.get("current_run_dir"),
                "ml_curation",
                status="running",
                detail="ML curation started",
            )
            with st.status(
                "Performing ML curation...",
                expanded=True,
            ) as status:
                ml_curator = cast("MLCurator | None", st.session_state.get("ml_curator"))
                if ml_curator is None:
                    ml_curator = MLCurator()
                    ml_curator.load_models(vertex_model_file, edge_model_file)

                if ml_curator.vertex_classifier is None or ml_curator.edge_classifier is None:
                    st.error("❌ ML models not loaded or trained. Cannot perform ML curation.")
                    _update_run_task(
                        st.session_state.get("current_run_dir"),
                        "ml_curation",
                        status="failed",
                        detail="ML models were not available for curation",
                    )
                    st.stop()

                curated_vertices = ml_curator.curate_vertices(
                    results["vertices"],
                    results["energy_data"],
                    st.session_state["image_shape"],
                    vertex_confidence_threshold,
                )

                curated_edges = ml_curator.curate_edges(
                    results["edges"],
                    curated_vertices,
                    results["energy_data"],
                    edge_confidence_threshold,
                )

                status.update(label="Rebuilding network after curation...", state="running")
                try:
                    baseline_counts, current_counts = _apply_curated_results(
                        curated_vertices,
                        curated_edges,
                        curation_mode="Machine Learning (Model-based)",
                    )
                except Exception as exc:
                    _update_run_task(
                        st.session_state.get("current_run_dir"),
                        "ml_curation",
                        status="failed",
                        detail=f"ML curation could not rebuild the network: {exc!s}",
                    )
                    st.error(
                        "Curated vertices and edges were not applied because the network "
                        f"could not be rebuilt: {exc!s}"
                    )
                    st.stop()

                st.success("✅ ML curation complete!")
                status.update(label="ML curation complete!", state="complete")
                _update_run_task(
                    st.session_state.get("current_run_dir"),
                    "ml_curation",
                    status="completed",
                    detail="ML curation complete and network rebuilt",
                )
                st.caption(
                    "The downstream network, exports, and share report now use the curated "
                    "vertices and edges."
                )

                col1, col2 = st.columns(2, gap="small")
                with col1:
                    st.metric(
                        "Vertices",
                        current_counts["Vertices"],
                        delta=current_counts["Vertices"] - baseline_counts["Vertices"],
                        help="Change relative to the pre-curation baseline",
                    )
                with col2:
                    st.metric(
                        "Edges",
                        current_counts["Edges"],
                        delta=current_counts["Edges"] - baseline_counts["Edges"],
                        help="Change relative to the pre-curation baseline",
                    )

    # Curation results
    if st.button("📊 Show Curation Statistics", width=250):
        st.markdown("### 📈 Curation Results")

        baseline_counts = st.session_state.get("curation_baseline_counts")
        if baseline_counts is None:
            st.info(
                "No curation has been applied yet. Run a curation step to compare before/after counts."
            )
        else:
            current_counts = summarize_processing_counts(st.session_state["processing_results"])
            curation_stats = pd.DataFrame(
                build_curation_stats_rows(baseline_counts, current_counts)
            )
            if curation_mode := st.session_state.get("last_curation_mode"):
                st.caption(
                    f"Most recent curation mode: {curation_mode}. "
                    "The network was rebuilt after the curated vertices and edges were applied."
                )

            st.dataframe(curation_stats, use_container_width=True)

            fig = px.bar(
                curation_stats,
                x="Component",
                y=["Original", "Current"],
                title="Curation Results",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)


def show_visualization_page():
    """Display the visualization page"""

    st.markdown('<h2 class="section-header">Network Visualization</h2>', unsafe_allow_html=True)

    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return

    st.markdown("""
    Visualize the vectorized vascular network in 2D and 3D. This section provides interactive tools to explore the results.
    Corresponds to `Visual` and `SpecialOutput` parameters in MATLAB.
    """)

    results = st.session_state["processing_results"]

    available_viz = []
    if "energy_data" in results:
        available_viz.append("Energy Field")
    if "vertices" in results and "edges" in results and "network" in results:
        available_viz.extend(["2D Network", "3D Network", "Depth Projection", "Strand Analysis"])

    if not available_viz:
        st.warning("⚠️ No visualizable results found in the current run.")
        return

    # Visualization options
    viz_type = st.selectbox(
        "Visualization type",
        available_viz,
        help="Choose the type of visualization to display",
    )

    col1, col2 = st.columns([3, 1], gap="large")

    with col2:
        st.markdown("### 🎨 Display Options")

        show_vertices = st.checkbox(
            "Show vertices", value=True, help="Display detected vertex markers"
        )
        show_edges = st.checkbox("Show edges", value=True, help="Display traced vessel segments")
        show_bifurcations = st.checkbox(
            "Show bifurcations", value=True, help="Highlight branching points in the network"
        )

        color_scheme = st.selectbox(
            "Color scheme",
            ["Energy", "Depth", "Strand ID", "Radius", "Length", "Random"],
            help="How to color the network components",
        )

        st.slider("Opacity", 0.1, 1.0, 0.8, 0.1, help="Adjust transparency of network rendering")

        if viz_type == "3D Network":
            st.selectbox(
                "Camera angle", ["Isometric", "Top", "Side", "Front"], help="3D viewing angle"
            )

    visualizer = NetworkVisualizer()

    with col1:
        st.markdown(f"### 📊 {viz_type}")

        # Generate actual visualization based on type
        if viz_type == "2D Network":
            fig = visualizer.plot_2d_network(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["processing_results"]["network"],
                st.session_state["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices,
                show_edges=show_edges,
                show_bifurcations=show_bifurcations,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "3D Network":
            fig = visualizer.plot_3d_network(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["processing_results"]["network"],
                st.session_state["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices,
                show_edges=show_edges,
                show_bifurcations=show_bifurcations,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Depth Projection":
            fig = visualizer.plot_depth_statistics(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["parameters"],
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Strand Analysis":
            fig = visualizer.plot_strand_analysis(
                st.session_state["processing_results"]["network"],
                st.session_state["processing_results"]["vertices"],
                st.session_state["parameters"],
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Energy Field":
            # For energy field, we need to pass the original image shape to the visualizer
            # and potentially allow selecting a slice axis and index
            st.info(
                "Energy Field visualization is a 2D slice. Select slice axis and index in sidebar."
            )
            slice_axis = st.sidebar.selectbox(
                "Slice Axis", [0, 1, 2], format_func=lambda x: ["Y", "X", "Z"][x]
            )
            energy = st.session_state["processing_results"]["energy_data"]["energy"]
            slice_index = st.sidebar.number_input(
                "Slice Index", value=int(energy.shape[slice_axis] // 2)
            )
            fig = visualizer.plot_energy_field(
                st.session_state["processing_results"]["energy_data"],
                slice_axis=slice_axis,
                slice_index=slice_index,
            )
            st.plotly_chart(fig, use_container_width=True)

    if not _has_full_network_results(results):
        st.info("Complete the full network stage to unlock exports and the share report.")
        return
    # Export options
    st.markdown("### 💾 Export Options")

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    # Prepare data for export if available
    vertices = st.session_state["processing_results"]["vertices"]
    edges = st.session_state["processing_results"]["edges"]
    network = st.session_state["processing_results"]["network"]
    parameters = st.session_state["processing_results"]["parameters"]
    current_run_dir = st.session_state.get("current_run_dir")

    for column, export_spec in zip((col1, col2, col3), EXPORT_BUTTON_SPECS):
        _render_export_download(
            column,
            run_dir=current_run_dir,
            vertices=vertices,
            edges=edges,
            network=network,
            parameters=parameters,
            export_spec=export_spec,
        )
    share_report_data = generate_share_report_data(
        st.session_state["processing_results"],
        st.session_state.get("dataset_name", "SLAVV dataset"),
        st.session_state.get("image_shape", (100, 100, 50)),
    )
    _log_share_report_prepared_once(
        st.session_state.get("dataset_name", "SLAVV dataset"),
        share_report_data,
        st.session_state["processing_results"],
    )
    _update_run_task(
        st.session_state.get("current_run_dir"),
        "share_report",
        status="completed",
        detail="Share report generated in app",
        artifacts={
            "share_report_file": share_report_data["file_name"],
            "share_report_signature": share_report_data["signature"],
        },
    )

    with col4:
        downloaded = st.download_button(
            label="Download Share Report",
            data=share_report_data["html"],
            file_name=share_report_data["file_name"],
            mime="text/html",
            help="Download a self-contained HTML report to share with collaborators.",
        )
        if downloaded:
            record_share_event(
                st.session_state,
                "share_report_downloaded",
                st.session_state.get("dataset_name", "SLAVV dataset"),
                share_report_data["signature"],
                extra={"report_file_name": share_report_data["file_name"]},
            )
            _update_run_task(
                st.session_state.get("current_run_dir"),
                "share_report",
                status="completed",
                detail="Share report downloaded",
                artifacts={"downloaded_report": share_report_data["file_name"]},
            )

    if downloaded:
        st.success(
            "Share report downloaded. Forward the HTML file to collaborators for offline review."
        )

    share_metrics = st.session_state.get("share_report_metrics", {})
    st.caption(
        "Tracked share events this session: "
        f"requested={share_metrics.get('share_report_requested', 0)}, "
        f"downloaded={share_metrics.get('share_report_downloaded', 0)}"
    )


def show_analysis_page():
    """Display the analysis page"""

    st.markdown('<h2 class="section-header">Network Analysis</h2>', unsafe_allow_html=True)

    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return

    results = st.session_state["processing_results"]
    if "network" not in results:
        st.warning(
            "⚠️ Analysis requires complete network extraction. Please run the pipeline up to the 'network' target."
        )
        return

    st.markdown("""
    Perform comprehensive statistical analysis on the vectorized vascular network. This section provides key metrics and detailed distributions.
    Corresponds to `SpecialOutput` parameters like `histograms`, `depth-stats`, `original-stats` in MATLAB.
    """)

    results = st.session_state["processing_results"]
    parameters = st.session_state["parameters"]
    _update_run_task(
        st.session_state.get("current_run_dir"),
        "analysis",
        status="completed",
        detail="Analysis dashboard viewed",
    )

    stats = compute_shareable_stats(
        results,
        image_shape=st.session_state.get("image_shape", (100, 100, 50)),
    )

    # Key metrics
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
    with col1:
        st.metric(
            "Total Length",
            f"{stats['total_length']:.1f} μm",
            help="Sum of all edge lengths",
        )
    with col2:
        st.metric(
            "Volume Fraction",
            f"{stats['volume_fraction']:.3f}",
            help="Fraction of volume occupied by vessels",
        )

    with col3:
        st.metric(
            "Bifurcation Density",
            f"{stats.get('bifurcation_density', 0):.2f} /mm³",
            help="Bifurcations per cubic millimeter",
        )

    with col4:
        st.metric(
            "Mean Radius",
            f"{stats.get('mean_radius', 0):.2f} μm",
            help="Average vessel radius",
        )
    # Detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🌳 Topology", "📏 Morphometry", "📊 Statistics"]
    )

    visualizer = NetworkVisualizer()

    with tab1:
        st.markdown("#### Length and Radius Distributions")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # Length distribution
            fig_length = visualizer.plot_strand_analysis(
                results["network"], results["vertices"], parameters
            )
            st.plotly_chart(fig_length, use_container_width=True)

        with col2:
            # Radius distribution
            fig_radius = visualizer.plot_radius_distribution(results["vertices"])
            st.plotly_chart(fig_radius, use_container_width=True)

        # Length-weighted histograms (ported from area_histogram_plotter.m)
        st.markdown("#### Length-Weighted Histograms")
        st.caption(
            "Depth, radius, and inclination distributions weighted by segment length. "
            "Ported from `area_histogram_plotter.m`."
        )

        try:
            fig_hist = visualizer.plot_length_weighted_histograms(
                results.get("vertices", {}),
                results.get("edges", {}),
                results.get("parameters", {}),
                number_of_bins=50,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.info(f"Length-weighted histograms unavailable: {e}")

    with tab2:
        st.markdown("#### Network Topology")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # Degree distribution
            fig_degree = visualizer.plot_degree_distribution(results["network"])
            st.plotly_chart(fig_degree, use_container_width=True)

        with col2:
            # Connectivity analysis
            connectivity_stats = pd.DataFrame(
                {
                    "Metric": [
                        "Connected Components",
                        "Average Path Length",
                        "Clustering Coefficient",
                        "Network Diameter",
                    ],
                    "Value": [
                        stats.get("num_connected_components", 0),
                        stats.get("avg_path_length", 0.0),
                        stats.get("clustering_coefficient", 0.0),
                        stats.get("network_diameter", 0.0),
                    ],
                }
            )
            st.dataframe(connectivity_stats, use_container_width=True)

    with tab3:
        st.markdown("#### Morphometric Analysis")

        # Depth-resolved statistics
        fig_depth = visualizer.plot_depth_statistics(
            results["vertices"], results["edges"], parameters
        )
        st.plotly_chart(fig_depth, use_container_width=True)

        # Tortuosity analysis
        col1, col2 = st.columns(2, gap="small")

        with col1:
            st.metric(
                "Mean Tortuosity",
                f"{stats.get('mean_tortuosity', 0):.2f}",
                help="Average path tortuosity",
            )
            st.metric(
                "Tortuosity Std",
                f"{stats.get('tortuosity_std', 0):.2f}",
                help="Standard deviation of tortuosity",
            )

        with col2:
            st.metric(
                "Fractal Dimension",
                f"{stats.get('fractal_dimension', 0):.2f}",
                help="Complexity of network structure",
            )
            st.metric(
                "Lacunarity",
                f"{stats.get('lacunarity', 0):.2f}",
                help="Spatial heterogeneity of the network",
            )

    with tab4:
        st.markdown("#### Complete Statistics Table")

        # Comprehensive statistics table
        full_stats = pd.DataFrame(
            {
                "Metric": [
                    "Number of Strands",
                    "Number of Bifurcations",
                    "Number of Endpoints",
                    "Total Length (μm)",
                    "Mean Strand Length (μm)",
                    "Length Density (μm/μm³)",
                    "Volume Fraction",
                    "Mean Radius (μm)",
                    "Radius Std (μm)",
                    "Bifurcation Density (/mm³)",
                    "Surface Area (μm²)",
                    "Mean Tortuosity",
                    "Number of Connected Components",
                    "Average Path Length",
                    "Clustering Coefficient",
                    "Network Diameter",
                    "Fractal Dimension",
                    "Lacunarity",
                    "Tortuosity Std",
                ],
                "Value": [
                    stats.get("num_strands", 0),
                    stats.get("num_bifurcations", 0),
                    stats.get("num_endpoints", 0),
                    f"{stats.get('total_length', 0):.1f}",
                    f"{stats.get('mean_strand_length', 0):.1f}",
                    f"{stats.get('length_density', 0):.3f}",
                    f"{stats.get('volume_fraction', 0):.4f}",
                    f"{stats.get('mean_radius', 0):.2f}",
                    f"{stats.get('radius_std', 0):.2f}",
                    f"{stats.get('bifurcation_density', 0):.2f}",
                    f"{stats.get('surface_area', 0):.1f}",
                    f"{stats.get('mean_tortuosity', 0):.3f}",
                    stats.get("num_connected_components", 0),
                    f"{stats.get('avg_path_length', 0):.2f}",
                    f"{stats.get('clustering_coefficient', 0):.2f}",
                    f"{stats.get('network_diameter', 0):.2f}",
                    f"{stats.get('fractal_dimension', 0):.2f}",
                    f"{stats.get('lacunarity', 0):.2f}",
                    f"{stats.get('tortuosity_std', 0):.2f}",
                ],
            }
        )

        st.dataframe(
            full_stats,
            use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", help="Statistic name"),
                "Value": st.column_config.TextColumn("Value", help="Computed value"),
            },
        )

        # Download statistics
        csv = full_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv,
            file_name="network_statistics.csv",
            mime="text/csv",
        )


def show_about_page():
    """Display the about page with detailed information about SLAVV."""
    st.markdown('<h2 class="section-header">About SLAVV</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Scientific Background

    SLAVV (Segmentation-Less, Automated, Vascular Vectorization) was developed to address the
    challenges of extracting vascular networks from large-scale microscopy volumes without
    requiring manual or error-prone segmentation steps.

    The algorithm uses a **multi-scale energy field** approach, where vessel centerlines are
    detected as local energy minima. This allows it to handle varying vessel diameters and
    low-contrast regions more robustly than threshold-based methods.

    ### 👨‍💻 Implementation Details

    This system is a modern Python implementation of the original SLAVV algorithm. Key improvements include:
    - **Performance**: Numba acceleration and multi-threaded processing.
    - **Scalability**: Chunk-based processing for large volumes.
    - **Modern UI**: Interactive Streamlit interface for parameter tuning and visualization.
    - **ML Integration**: Machine learning models for automatic quality control.

    ### 📜 Credits and License

    - **Original Algorithm**: Samuel Alexander Mihelic
    - **Python Port**: Developed for modern high-throughput analysis.
    - **License**: Provided as open-source for scientific research.

    For more information or to cite this work, please refer to the project documentation.
    """)
