"""Application shell, navigation, and shared sidebar context."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from slavv_python.interface.streamlit.services.host_paths import (
    file_manager_action_label,
    reveal_run_directory,
)
from slavv_python.interface.streamlit.services.run_monitor import render_sidebar_run_pulse

from .navigation import register_pages, switch_to
from .state.workflow import install_loaded_run, load_persisted_run, summarize_workflow
from .state.workspace_view import workspace_view_from_session
from .state.workspaces import DEFAULT_WORKSPACE_ROOTS, discover_workspaces

PAGE_HANDLERS = {
    "dashboard": "routes/dashboard.py",
    "workspaces": "routes/workspaces.py",
    "processing": "routes/processing.py",
    "curation": "routes/curation.py",
    "visualization": "routes/visualization.py",
    "analysis": "routes/analysis.py",
    "about": "routes/about.py",
}


def _render_run_loader() -> None:
    """Render the read-only persisted-run loader."""
    with st.sidebar.popover("Open existing run", icon=":material/folder_open:", width="stretch"):
        st.caption(
            "Load compatible checkpoints into this session. The source directory remains read-only."
        )
        quick_records = discover_workspaces(
            roots=DEFAULT_WORKSPACE_ROOTS,
            active_run_dir=st.session_state.get("current_run_dir"),
            limit=12,
        )
        quick_paths = [record.run_dir for record in quick_records if record.loadable]
        if quick_paths:
            quick_pick = st.selectbox(
                "Recent workspace runs",
                options=["", *quick_paths],
                format_func=lambda value: (
                    "Type or browse a path below"
                    if not value
                    else next(
                        (record.name for record in quick_records if record.run_dir == value),
                        Path(value).name,
                    )
                ),
                key="run_path_quick_pick",
            )
            if quick_pick:
                st.session_state["run_path_input"] = quick_pick
        default_root = next(
            (str(path) for _label, path in DEFAULT_WORKSPACE_ROOTS if path.is_dir()),
            "",
        )
        with st.form("open_existing_run_form"):
            run_path = st.text_input(
                "Run directory",
                value=str(st.session_state.get("run_path_input", default_root)),
                placeholder=str(Path(default_root) / "oracle_180709_E" / "crop_M_exact_v3")
                if default_root
                else "/path/to/structured_run",
            )
            submitted = st.form_submit_button("Open run", type="primary")
        if submitted:
            result = load_persisted_run(run_path)
            if not result.ok:
                st.error(result.error or "The run could not be opened.")
            else:
                install_loaded_run(st.session_state, result)
                st.session_state["run_path_input"] = run_path
                st.toast(
                    "Loaded " + ", ".join(stage.title() for stage in result.loaded_stages),
                    icon=":material/task_alt:",
                )
                st.rerun()


def _render_sidebar_context() -> None:
    """Show shared dataset and workflow status beneath primary navigation."""
    summary = summarize_workflow(st.session_state)
    workspace = workspace_view_from_session(st.session_state, summary=summary)
    st.sidebar.divider()
    st.sidebar.caption("CURRENT WORKSPACE")
    st.sidebar.markdown(f"**{summary.dataset_name}**")
    source_label = {
        "live": "Live processing run",
        "reopened": "Reopened run",
        "empty": "No active run",
    }.get(summary.source_kind, summary.source_kind.title())
    badge_color = "green" if summary.ready_stages else "gray"
    st.sidebar.badge(source_label, color=badge_color)
    if summary.read_only and summary.ready_stages:
        st.sidebar.badge("Read-only source", icon=":material/lock:", color="orange")

    stage_labels = {
        "energy": "Energy",
        "vertices": "Vertices",
        "edges": "Edges",
        "network": "Network",
    }
    stage_line = " · ".join(
        f"{'✓' if stage in summary.ready_stages else '○'} {label}"
        for stage, label in stage_labels.items()
    )
    st.sidebar.caption(stage_line)
    if summary.curation_mode:
        st.sidebar.caption(f"Last curation: {summary.curation_mode}")

    if st.sidebar.button(
        summary.next_label,
        type="primary",
        icon=":material/arrow_forward:",
        width="stretch",
    ):
        switch_to(summary.next_page)

    _render_run_loader()
    if workspace.has_run:
        render_sidebar_run_pulse(
            workspace.run_dir,
            snapshot=workspace.snapshot,
        )
        with st.sidebar.expander("Run location"):
            st.code(workspace.run_dir, language=None)
            reveal_label = file_manager_action_label()
            if reveal_label is not None:
                if st.button(reveal_label, icon=":material/folder:", width="stretch"):
                    try:
                        reveal_run_directory(workspace.run_dir)
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
            else:
                st.caption("Copy the path above to open it on this host.")


def main() -> None:
    """Run the fully connected Streamlit workspace."""
    with st.sidebar:
        st.title("SLAVV")
        st.caption("Paper Path GUI · monitor Exact Route parity runs in Workspaces")
        st.caption("From 3D TIFF to curated vascular network")
    navigation = st.navigation(register_pages(PAGE_HANDLERS), position="sidebar", expanded=True)
    _render_sidebar_context()
    navigation.run()


__all__ = ["PAGE_HANDLERS", "main"]
