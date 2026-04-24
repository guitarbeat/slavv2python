"""Focused app-facing helpers for dashboard rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from source.runtime.run_state import target_stage_progress

from .web_app_dashboard import DASHBOARD_STAGE_ORDER


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


__all__ = ["render_run_dashboard"]


