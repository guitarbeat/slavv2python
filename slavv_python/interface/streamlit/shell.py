"""Application shell, navigation, and shared sidebar context."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from .navigation import register_pages, switch_to
from .state.workflow import install_loaded_run, load_persisted_run, summarize_workflow

PAGE_HANDLERS = {
    "dashboard": "routes/dashboard.py",
    "workspaces": "routes/workspaces.py",
    "processing": "routes/processing.py",
    "curation": "routes/curation.py",
    "visualization": "routes/visualization.py",
    "analysis": "routes/analysis.py",
    "about": "routes/about.py",
}


def _open_run_folder(run_dir: str) -> None:
    """Open a validated run directory in Windows Explorer."""
    path = Path(run_dir).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("The active run directory is unavailable.")
    subprocess.Popen(["explorer.exe", str(path)], close_fds=True)


def _render_run_loader() -> None:
    """Render the read-only persisted-run loader."""
    with st.sidebar.popover("Open existing run", icon=":material/folder_open:", width="stretch"):
        st.caption(
            "Load compatible checkpoints into this session. The source directory remains read-only."
        )
        with st.form("open_existing_run_form"):
            run_path = st.text_input(
                "Run directory",
                value=str(st.session_state.get("run_path_input", "")),
                placeholder=r"D:\path\to\structured_run",
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
    if summary.run_dir:
        with st.sidebar.expander("Run location"):
            st.code(summary.run_dir, language=None)
            if sys.platform == "win32":
                if st.button("Open in Explorer", icon=":material/folder:", width="stretch"):
                    try:
                        _open_run_folder(summary.run_dir)
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
            else:
                st.caption("Copy the path above to open it on this host.")


def main() -> None:
    """Run the fully connected Streamlit workspace."""
    with st.sidebar:
        st.title("SLAVV")
        st.caption("From 3D TIFF to curated vascular network")
    navigation = st.navigation(register_pages(PAGE_HANDLERS), position="sidebar", expanded=True)
    _render_sidebar_context()
    navigation.run()


__all__ = ["PAGE_HANDLERS", "main"]
