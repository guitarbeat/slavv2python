"""Workspace explorer for inspecting and reopening structured SLAVV runs."""

from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from slavv_python.interface.streamlit.navigation import switch_to
from slavv_python.interface.streamlit.state.workflow import install_loaded_run, load_persisted_run
from slavv_python.interface.streamlit.state.workspaces import WorkspaceRecord, discover_workspaces


@st.cache_data(ttl=5, show_spinner=False)
def _cached_workspaces(active_run_dir: str | None) -> tuple[WorkspaceRecord, ...]:
    return discover_workspaces(active_run_dir=active_run_dir)


def _install_workspace_styles() -> None:
    """Install restrained state colors and page-level interaction polish."""
    st.html(
        """
        <style>
            @keyframes ws-enter {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .ws-summary {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin: 1.25rem 0 1.75rem;
                border-top: 1px solid rgba(22, 60, 56, 0.18);
                border-bottom: 1px solid rgba(22, 60, 56, 0.18);
                animation: ws-enter 220ms ease-out both;
            }
            .ws-summary-item { padding: 0.9rem 1rem 0.85rem 0; }
            .ws-summary-item + .ws-summary-item {
                border-left: 1px solid rgba(22, 60, 56, 0.12);
                padding-left: 1rem;
            }
            .ws-summary-label {
                color: #526c68 !important;
                font-size: 0.76rem;
                font-weight: 650;
                letter-spacing: 0.055em;
                text-transform: uppercase;
            }
            .ws-summary-value {
                color: #0b3934 !important;
                font-size: 1.65rem;
                font-weight: 690;
                letter-spacing: -0.035em;
                line-height: 1.2;
                margin-top: 0.15rem;
            }
            .ws-summary-item.complete { border-top: 3px solid #18856f; }
            .ws-summary-item.partial { border-top: 3px solid #d68a25; }
            .ws-summary-item.running { border-top: 3px solid #2f6fdb; }
            .ws-summary-item.active { border-top: 3px solid #795bbd; }
            .ws-stage-track {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.55rem;
                margin: 0.65rem 0 1.1rem;
                animation: ws-enter 260ms 40ms ease-out both;
            }
            .ws-stage {
                border: 1px solid rgba(22, 60, 56, 0.14);
                border-radius: 0.5rem;
                padding: 0.6rem 0.7rem;
                transition: transform 140ms ease, border-color 140ms ease;
            }
            .ws-stage:hover { transform: translateY(-1px); }
            .ws-stage.ready { background: #eaf7f3; border-color: #8bc7b8; color: #11604f; }
            .ws-stage.running { background: #eaf1fd; border-color: #96b5ec; color: #2459a9; }
            .ws-stage.missing { background: #f4f6f6; color: #71807e; }
            .ws-stage.error { background: #fcebed; border-color: #e3a3aa; color: #9a3340; }
            .ws-stage-name { font-weight: 660; }
            .ws-stage-state { font-size: 0.76rem; margin-top: 0.12rem; opacity: 0.82; }
            @media (max-width: 760px) {
                .ws-summary, .ws-stage-track { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (prefers-color-scheme: dark) {
                .ws-summary-value { color: #e3efed; }
                .ws-summary-label { color: #a9c3bf; }
                .ws-stage.ready { background: #123e35; color: #a9e4d5; }
                .ws-stage.running { background: #172f56; color: #bdd2f6; }
                .ws-stage.missing { background: #252a2a; color: #a9b2b1; }
                .ws-stage.error { background: #4b2228; color: #f0b9c0; }
            }
        </style>
        """
    )


def _workspace_state(record: WorkspaceRecord) -> str:
    if "fail" in record.status or "error" in record.status:
        return "Errors"
    if "running" in record.status or "pending" in record.status:
        return "Running"
    if "network" in record.ready_stages and record.status.startswith("completed"):
        return "Complete"
    return "Partial"


def _workspace_table(records: tuple[WorkspaceRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Workspace": f"● {record.name}" if record.is_active else record.name,
                "State": _workspace_state(record),
                "Progress": record.progress,
                "Through": (
                    record.ready_stages[-1].title() if record.ready_stages else "Metadata only"
                ),
                "Updated": record.updated_at.replace("T", " ").replace("Z", " UTC"),
            }
            for record in records
        ]
    )


def _open_folder(run_dir: str) -> None:
    path = Path(run_dir).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("The workspace directory is unavailable.")
    subprocess.Popen(["explorer.exe", str(path)], close_fds=True)


def _render_summary(records: tuple[WorkspaceRecord, ...]) -> None:
    complete = sum(_workspace_state(record) == "Complete" for record in records)
    partial = sum(_workspace_state(record) == "Partial" for record in records)
    running = sum(_workspace_state(record) == "Running" for record in records)
    active = sum(record.is_active for record in records)
    st.html(
        f"""
        <div class="ws-summary">
            <div class="ws-summary-item complete">
                <div class="ws-summary-label">Complete networks</div>
                <div class="ws-summary-value">{complete}</div>
            </div>
            <div class="ws-summary-item partial">
                <div class="ws-summary-label">Partial runs</div>
                <div class="ws-summary-value">{partial}</div>
            </div>
            <div class="ws-summary-item running">
                <div class="ws-summary-label">Running now</div>
                <div class="ws-summary-value">{running}</div>
            </div>
            <div class="ws-summary-item active">
                <div class="ws-summary-label">Active session</div>
                <div class="ws-summary-value">{active or 'None'}</div>
            </div>
        </div>
        """
    )


def _render_stage_track(record: WorkspaceRecord) -> None:
    stage_blocks: list[str] = []
    for stage in record.stages:
        if "fail" in stage.status or "error" in stage.status:
            css_state = "error"
        elif "running" in stage.status:
            css_state = "running"
        elif stage.name in record.ready_stages:
            css_state = "ready"
        else:
            css_state = "missing"
        label = "Ready" if css_state == "ready" else stage.status.replace("_", " ").title()
        stage_blocks.append(
            f'<div class="ws-stage {css_state}">'
            f'<div class="ws-stage-name">{html.escape(stage.name.title())}</div>'
            f'<div class="ws-stage-state">{html.escape(label)}</div>'
            "</div>"
        )
    st.html(f'<div class="ws-stage-track">{"".join(stage_blocks)}</div>')


def _render_stage_details(record: WorkspaceRecord) -> None:
    stage_frame = pd.DataFrame(
        [
            {
                "Stage": stage.name.title(),
                "Status": stage.status.replace("_", " ").title(),
                "Progress": stage.progress,
                "Elapsed (s)": round(stage.elapsed_seconds, 2),
                "Last detail": stage.detail or "—",
            }
            for stage in record.stages
        ]
    )
    st.dataframe(
        stage_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Stage": st.column_config.TextColumn(width="small"),
            "Status": st.column_config.TextColumn(width="small"),
            "Progress": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
            "Elapsed (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Last detail": st.column_config.TextColumn(width="large"),
        },
    )


def _render_active_actions(record: WorkspaceRecord) -> None:
    actions = st.columns(4, gap="small")
    if actions[0].button("Processing", icon=":material/tune:", width="stretch"):
        switch_to("processing")
    if actions[1].button(
        "Curation",
        icon=":material/edit_note:",
        width="stretch",
        disabled="edges" not in record.ready_stages,
    ):
        switch_to("curation")
    if actions[2].button(
        "Visualization",
        icon=":material/view_in_ar:",
        width="stretch",
        disabled="network" not in record.ready_stages,
    ):
        switch_to("visualization")
    if actions[3].button(
        "Analysis",
        icon=":material/analytics:",
        width="stretch",
        disabled="network" not in record.ready_stages,
    ):
        switch_to("analysis")


def _render_inspector(record: WorkspaceRecord) -> None:
    heading, location = st.columns([3, 2], gap="large", vertical_alignment="bottom")
    with heading:
        st.subheader(record.name)
        state = _workspace_state(record)
        color = {"Complete": "green", "Partial": "orange", "Running": "blue", "Errors": "red"}[
            state
        ]
        st.badge(state, color=color)
        if record.is_active:
            st.badge("Active session", icon=":material/radio_button_checked:", color="blue")
        if record.error_count:
            st.badge(
                f"{record.error_count} recorded error{'s' if record.error_count != 1 else ''}",
                icon=":material/warning:",
                color="red",
            )
    with location:
        st.caption(f"{record.source} · updated {record.updated_at.replace('T', ' ')[:19]}")

    _render_stage_track(record)
    facts = st.columns(4, gap="small")
    facts[0].metric("Run ID", record.run_id or "Not recorded")
    facts[1].metric("Requested through", record.target_stage.title())
    facts[2].metric(
        "Volume shape",
        " x ".join(str(value) for value in record.image_shape)
        if record.image_shape
        else "Not recorded",
    )
    facts[3].metric("Pipeline progress", f"{record.progress:.0%}")

    detail_tab, files_tab = st.tabs(["Stage details", "Run location"])
    with detail_tab:
        _render_stage_details(record)
    with files_tab:
        st.caption("The source remains unchanged when opened in this application.")
        st.code(record.run_dir, language=None)
        if sys.platform == "win32" and st.button(
            "Open in Explorer",
            icon=":material/folder:",
            key="workspace_open_explorer",
        ):
            try:
                _open_folder(record.run_dir)
            except (OSError, ValueError) as exc:
                st.error(str(exc))

    if record.is_active:
        _render_active_actions(record)
        return

    if st.button(
        "Open workspace read-only",
        type="primary",
        icon=":material/folder_open:",
        width="stretch",
        disabled=not record.loadable,
        help=(
            "Load available typed stage results into this browser session. Source files are unchanged."
            if record.loadable
            else "This workspace does not contain validated settings and a stage result."
        ),
    ):
        with st.spinner("Loading workspace stage results..."):
            result = load_persisted_run(record.run_dir)
        if result.ok:
            install_loaded_run(st.session_state, result)
            st.session_state["dataset_name"] = record.name
            st.toast(f"Opened {record.name}", icon=":material/task_alt:")
            st.rerun()
        else:
            st.error(result.error or "The workspace could not be opened.")


def _render_comparison(records: tuple[WorkspaceRecord, ...]) -> None:
    with st.expander("Compare workspaces", icon=":material/compare_arrows:"):
        paths = [record.run_dir for record in records]
        record_by_path = {record.run_dir: record for record in records}
        selected = st.multiselect(
            "Choose up to four runs",
            paths,
            max_selections=4,
            format_func=lambda path: record_by_path[path].name,
            key="workspace_comparison",
        )
        if not selected:
            st.caption("Choose runs to compare progress, stage readiness, and volume shape.")
            return
        comparison = pd.DataFrame(
            [
                {
                    "Workspace": record_by_path[path].name,
                    "State": _workspace_state(record_by_path[path]),
                    "Progress": record_by_path[path].progress,
                    "Ready stages": len(record_by_path[path].ready_stages),
                    "Target": record_by_path[path].target_stage.title(),
                    "Volume": (
                        " x ".join(str(value) for value in record_by_path[path].image_shape)
                        if record_by_path[path].image_shape
                        else "Not recorded"
                    ),
                }
                for path in selected
            ]
        )
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "Progress": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
                "Ready stages": st.column_config.NumberColumn(format="%d / 4"),
            },
        )


def show_workspaces_page() -> None:
    """Display searchable workspace inventory and selected-run metadata."""
    _install_workspace_styles()
    st.header("Workspace explorer")
    st.caption("Find, compare, inspect, and safely reopen pipeline runs.")

    active_run_dir = str(st.session_state.get("current_run_dir") or "") or None
    records = _cached_workspaces(active_run_dir)

    with st.sidebar:
        st.divider()
        st.subheader("Workspace filters")
        query = st.text_input(
            "Search",
            placeholder="Name, run ID, or path",
            key="workspace_search",
        ).strip().casefold()
        state_filter = (
            st.segmented_control(
                "Run state",
                ("All", "Complete", "Partial", "Running", "Errors"),
                default="All",
                key="workspace_state_filter",
            )
            or "All"
        )
        source_options = tuple(sorted({record.source for record in records}))
        selected_sources = st.multiselect(
            "Locations",
            source_options,
            default=source_options,
            key="workspace_sources",
        )
        sort_by = st.selectbox(
            "Sort by",
            ("Newest first", "Name", "Most complete"),
            key="workspace_sort",
        )
        only_loadable = st.toggle(
            "Only compatible runs",
            value=True,
            help="Require validated settings and at least one typed stage checkpoint.",
        )
        if st.button("Refresh list", icon=":material/refresh:", width="stretch"):
            _cached_workspaces.clear()
            st.rerun()

    filtered = tuple(
        record
        for record in records
        if record.source in selected_sources
        and (not only_loadable or record.loadable)
        and (state_filter == "All" or _workspace_state(record) == state_filter)
        and (
            not query
            or query in record.name.casefold()
            or query in record.run_id.casefold()
            or query in record.run_dir.casefold()
        )
    )
    if sort_by == "Name":
        filtered = tuple(sorted(filtered, key=lambda record: record.name.casefold()))
    elif sort_by == "Most complete":
        filtered = tuple(
            sorted(filtered, key=lambda record: (record.progress, record.updated_at), reverse=True)
        )

    _render_summary(filtered)
    if not filtered:
        st.info("No compatible workspaces match these filters.")
        return

    heading, count = st.columns([4, 1], vertical_alignment="bottom")
    heading.subheader("Saved runs")
    count.caption(f"{len(filtered)} shown")
    selection = st.dataframe(
        _workspace_table(filtered),
        hide_index=True,
        width="stretch",
        height=min(420, 38 + len(filtered) * 35),
        on_select="rerun",
        selection_mode="single-row",
        key="workspace_inventory",
        column_config={
            "Workspace": st.column_config.TextColumn(width="medium"),
            "State": st.column_config.TextColumn(width="small"),
            "Progress": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
            "Through": st.column_config.TextColumn(width="small"),
            "Updated": st.column_config.TextColumn(width="medium"),
        },
    )
    _render_comparison(filtered)
    selected_rows = selection.selection.rows
    selected_index = selected_rows[0] if selected_rows else 0
    st.divider()
    _render_inspector(filtered[selected_index])


__all__ = ["show_workspaces_page"]
