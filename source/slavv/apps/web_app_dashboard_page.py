from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import plotly.express as px
import streamlit as st

from slavv.runtime.run_state import target_stage_progress
from slavv.visualization import NetworkVisualizer

from .dashboard_state import DashboardContext, load_dashboard_context
from .web_app_dashboard import (
    DASHBOARD_BREAKDOWN_SECTIONS,
    DASHBOARD_PLACEHOLDER,
    DASHBOARD_STAGE_ORDER,
    _build_dashboard_placeholder_trend,
    _dashboard_breakdown_frame,
    _dashboard_stage_frame,
    build_dashboard_backlog_frame,
    filter_dashboard_breakdown,
)

if TYPE_CHECKING:
    from slavv.runtime import RunSnapshot

DASHBOARD_ASSUMPTION = (
    "Assumption: until dashboard metrics are specified, this view summarizes the active run, "
    "current network outputs, and share-report activity for the current session."
)
DASHBOARD_RELEASE_URL = "https://docs.streamlit.io/develop/quick-reference/release-notes"
DASHBOARD_REPO_URL = "https://github.com/UTFOIL/slavv2python"


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
    return load_dashboard_context(st.session_state)


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
