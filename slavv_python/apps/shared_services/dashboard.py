"""Focused app-facing helpers for dashboard rendering."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Any

from slavv_python.runtime.constants import TRACKED_RUN_STAGES
from slavv_python.runtime.status import target_stage_progress

DASHBOARD_STAGE_ORDER = TRACKED_RUN_STAGES
DASHBOARD_BREAKDOWN_SECTIONS = {
    "Pipeline": "Core processing stages and run state.",
    "Network": "Graph topology and extracted metrics.",
    "Activity": "Session-level events and share reports.",
}
DASHBOARD_PLACEHOLDER = "---"


def render_run_dashboard(snapshot) -> None:
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


def build_dashboard_placeholder_trend() -> go.Figure:
    """Return an empty plotly figure for trend slots awaiting data."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": "Awaiting data...",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 16}
        }]
    )
    return fig


def build_dashboard_stage_frame(snapshot: Any | None, run_dir: str | None = None) -> pd.DataFrame:
    """Build a DataFrame of stage-level progress for trend plotting."""
    rows = []
    for stage in DASHBOARD_STAGE_ORDER:
        progress = 0.0
        if snapshot and stage in snapshot.stages:
            progress = snapshot.stages[stage].progress
        rows.append({"Stage": stage, "Progress (%)": progress * 100})
    
    if not rows:
        return pd.DataFrame(columns=["Stage", "Progress (%)"])
    return pd.DataFrame(rows)


def build_dashboard_breakdown_frame(
    snapshot: Any | None,
    stats: dict[str, Any] | None,
    share_metrics: dict[str, Any],
    run_dir: str | None = None,
) -> pd.DataFrame:
    """Build a comprehensive breakdown of all dashboard metrics."""
    rows = []
    
    # 1. Pipeline Section
    for stage in DASHBOARD_STAGE_ORDER:
        progress = 0.0
        status = "pending"
        value = DASHBOARD_PLACEHOLDER
        if snapshot and stage in snapshot.stages:
            s = snapshot.stages[stage]
            progress = s.progress
            status = s.status
            value = f"{int(progress * 100)}%"
            
        rows.append({
            "Section": "Pipeline",
            "Metric": f"Stage: {stage}",
            "Progress": progress * 100,
            "Value": value,
            "Status": status,
            "Source": "Run Snapshot",
            "Notes": f"Tracked progress for {stage}."
        })

    # 2. Network Section
    network_metrics = [
        ("Vertices", "num_vertices", "count"),
        ("Edges", "num_edges", "count"),
        ("Strands", "num_strands", "count"),
        ("Total Length", "total_length", "um"),
    ]
    for label, key, unit in network_metrics:
        val = stats.get(key) if stats else None
        rows.append({
            "Section": "Network",
            "Metric": label,
            "Progress": 100 if val is not None else 0,
            "Value": f"{val:.1f} {unit}" if isinstance(val, (int, float)) else DASHBOARD_PLACEHOLDER,
            "Status": "ready" if val is not None else "awaiting",
            "Source": "Network Stats",
            "Notes": f"Extracted {label.lower()} from result graph."
        })

    # 3. Activity Section
    rows.append({
        "Section": "Activity",
        "Metric": "Share Reports",
        "Progress": 0,
        "Value": str(share_metrics.get("share_report_requested", 0)),
        "Status": "active",
        "Source": "Session State",
        "Notes": "Total share reports generated in this session."
    })

    return pd.DataFrame(rows)


def filter_dashboard_breakdown(
    frame: pd.DataFrame,
    focus: str = "Overview",
    selected_sections: list[str] | None = None,
    show_placeholders: bool = True,
) -> pd.DataFrame:
    """Filter the breakdown frame based on UI focus and section selection."""
    if frame.empty:
        return frame
    
    df = frame.copy()
    
    if focus == "Pipeline":
        df = df[df["Section"] == "Pipeline"]
    elif focus == "Network":
        df = df[df["Section"] == "Network"]
        
    if selected_sections:
        df = df[df["Section"].isin(selected_sections)]
        
    if not show_placeholders:
        df = df[df["Value"] != DASHBOARD_PLACEHOLDER]
        
    return df


def build_dashboard_backlog_frame(
    requests: list[dict[str, Any]],
    repo_url: str = "",
    release_url: str = "",
) -> pd.DataFrame:
    """Build a DataFrame for the dashboard extension backlog."""
    if not requests:
        # Return empty frame with correct columns for Streamlit column_config
        return pd.DataFrame(columns=[
            "Metric", "Owner", "Priority", "Tracked", "Status", "Reference", "Notes"
        ])
    
    rows = []
    for req in requests:
        rows.append({
            "Metric": req.get("metric", "Unknown"),
            "Owner": req.get("owner", "Pipeline"),
            "Priority": req.get("priority", "Medium"),
            "Tracked": False,
            "Status": "TODO",
            "Reference": repo_url,
            "Notes": req.get("notes", "")
        })
    return pd.DataFrame(rows)


__all__ = [
    "DASHBOARD_BREAKDOWN_SECTIONS",
    "DASHBOARD_PLACEHOLDER",
    "DASHBOARD_STAGE_ORDER",
    "build_dashboard_backlog_frame",
    "build_dashboard_breakdown_frame",
    "build_dashboard_placeholder_trend",
    "build_dashboard_stage_frame",
    "filter_dashboard_breakdown",
    "render_run_dashboard",
]
