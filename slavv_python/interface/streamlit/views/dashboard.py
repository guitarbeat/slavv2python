"""Live workflow dashboard for the connected Streamlit application."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import streamlit as st

from slavv_python.interface.streamlit.navigation import switch_to
from slavv_python.interface.streamlit.services.run_monitor import (
    render_run_ops_panel,
    render_stage_unit_bars,
)
from slavv_python.interface.streamlit.state.curation import summarize_processing_counts
from slavv_python.interface.streamlit.state.dashboard import load_dashboard_context
from slavv_python.interface.streamlit.state.workflow import STAGE_ORDER, summarize_workflow

DASHBOARD_REPO_URL = "https://github.com/UTFOIL/slavv2python"
DASHBOARD_DOCS_URL = f"{DASHBOARD_REPO_URL}/blob/main/docs/README.md"
DASHBOARD_CURATION_URL = (
    f"{DASHBOARD_REPO_URL}/blob/main/docs/reference/workflow/MANUAL_CURATION_WORKFLOW.md"
)


def _workflow_rows(ready_stages: tuple[str, ...]) -> list[dict[str, str]]:
    page_by_stage = {
        "energy": "Processing",
        "vertices": "Processing",
        "edges": "Curation",
        "network": "Visualization / Analysis",
    }
    return [
        {
            "Stage": stage.title(),
            "Status": "Ready" if stage in ready_stages else "Not available",
            "Used in": page_by_stage[stage],
        }
        for stage in STAGE_ORDER
    ]


def _render_headline_metrics(results: Any | None) -> None:
    if results is None:
        counts = dict.fromkeys(("Vertices", "Edges", "Strands", "Bifurcations"), 0)
    else:
        counts = summarize_processing_counts(results)
    columns = st.columns(4, gap="small")
    for column, label in zip(columns, ("Vertices", "Edges", "Strands", "Bifurcations")):
        column.metric(label, counts[label])


def show_dashboard_page() -> None:
    """Display live workflow state, readiness, and the next recommended action."""
    st.session_state.setdefault("dashboard_focus", "Overview")
    summary = summarize_workflow(st.session_state)
    context = load_dashboard_context(st.session_state)
    results = context["results"]
    snapshot = context["snapshot"]

    st.header("SLAVV workflow")
    st.caption("Process, review, visualize, and analyze one vascular dataset.")
    focus = st.segmented_control(
        "Dashboard focus",
        ("Overview", "Pipeline", "Network"),
        key="dashboard_focus",
        bind="query-params",
        label_visibility="collapsed",
    )

    status_col, action_col = st.columns([3, 1], gap="large", vertical_alignment="center")
    with status_col:
        st.subheader(summary.dataset_name)
        source = {
            "empty": "No active run",
            "live": "Live processing run",
            "reopened": "Reopened run",
        }.get(summary.source_kind, summary.source_kind.title())
        status_text = source + (" · read-only source" if summary.read_only else "")
        st.caption(status_text)
    with action_col:
        if st.button(
            summary.next_label,
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
        ):
            switch_to(summary.next_page)

    if not summary.ready_stages:
        st.info(
            "Process an uploaded TIFF, choose a built-in sample in Processing, or open "
            "an existing structured run."
        )

    if focus in (None, "Overview", "Network"):
        _render_headline_metrics(results)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Pipeline readiness")
        st.dataframe(
            pd.DataFrame(_workflow_rows(summary.ready_stages)),
            hide_index=True,
            width="stretch",
            column_config={
                "Stage": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Used in": st.column_config.TextColumn(width="medium"),
            },
        )
        if snapshot is not None and focus in (None, "Overview", "Pipeline"):
            st.caption(
                f"Saved run status: {snapshot.status} · requested through: "
                f"{snapshot.target_stage} · current stage: {snapshot.current_stage or 'complete'}"
            )
            if focus in (None, "Pipeline"):
                render_stage_unit_bars(snapshot)
            if context["run_dir"]:
                render_run_ops_panel(
                    context["run_dir"],
                    snapshot=snapshot,
                    expanded=focus in (None, "Pipeline"),
                )
    with right:
        st.subheader("Session activity")
        st.metric("Curation", summary.curation_mode or "Not applied")
        share_metrics = cast("dict[str, Any]", context["share_metrics"])
        st.metric(
            "Share reports",
            int(share_metrics.get("share_report_downloaded", 0)),
            help="Reports downloaded in this browser session.",
        )
        if summary.run_dir:
            st.caption("The full source path is available in the sidebar Run location section.")

    if st.session_state.get("curation_baseline_counts") and results is not None:
        baseline = st.session_state["curation_baseline_counts"]
        current = summarize_processing_counts(st.session_state["processing_results"])
        st.subheader("Latest curation change")
        curation_columns = st.columns(4, gap="small")
        for column, label in zip(
            curation_columns, ("Vertices", "Edges", "Strands", "Bifurcations")
        ):
            column.metric(label, current[label], delta=current[label] - baseline[label])

    st.divider()
    links = st.columns(3, gap="small")
    links[0].link_button("Documentation", DASHBOARD_DOCS_URL, width="stretch")
    links[1].link_button("Manual curation workflow", DASHBOARD_CURATION_URL, width="stretch")
    links[2].link_button("Repository", DASHBOARD_REPO_URL, width="stretch")


__all__ = ["DASHBOARD_REPO_URL", "show_dashboard_page"]
