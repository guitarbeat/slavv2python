from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from source.utils.formatting import format_time

DASHBOARD_STAGE_ORDER = ("energy", "vertices", "edges", "network")
DASHBOARD_PLACEHOLDER = "Awaiting data"
DASHBOARD_BREAKDOWN_SECTIONS = ("Pipeline", "Network", "Share Report", "Optional Tasks")


def _dashboard_snapshot_source(run_dir: str | None) -> str:
    """Return the most specific snapshot source label available."""
    if not run_dir:
        return "No run snapshot loaded"
    return str(Path(run_dir) / "99_Metadata" / "run_snapshot.json")


def _dashboard_stage_frame(snapshot: Any | None, run_dir: str | None = None) -> pd.DataFrame:
    """Return pipeline stage progress for dashboard charts and tables."""
    rows = []
    for stage_name in DASHBOARD_STAGE_ORDER:
        stage_snapshot = None if snapshot is None else snapshot.stages.get(stage_name)
        rows.append(
            {
                "Stage": stage_name.title(),
                "Progress (%)": int(stage_snapshot.progress * 100) if stage_snapshot else 0,
                "Status": stage_snapshot.status if stage_snapshot else "placeholder",
                "Detail": (
                    (stage_snapshot.detail or stage_snapshot.substage)
                    if stage_snapshot
                    else "Waiting for a processed run"
                ),
                "Source": "99_Metadata/run_snapshot.json"
                if snapshot is not None
                else _dashboard_snapshot_source(run_dir),
            }
        )
    return pd.DataFrame(rows)


def _dashboard_run_throughput_rows(
    snapshot: Any | None,
    run_dir: str | None = None,
) -> list[dict[str, object]]:
    """Build per-run throughput metrics from the persisted snapshot."""
    source = (
        "99_Metadata/run_snapshot.json"
        if snapshot is not None
        else _dashboard_snapshot_source(run_dir)
    )
    if snapshot is None:
        return [
            {
                "Section": "Pipeline",
                "Metric": "Elapsed runtime",
                "Progress": None,
                "Value": DASHBOARD_PLACEHOLDER,
                "Status": "placeholder",
                "Source": source,
                "Notes": "Available once a run snapshot is loaded.",
            },
            {
                "Section": "Pipeline",
                "Metric": "ETA",
                "Progress": None,
                "Value": DASHBOARD_PLACEHOLDER,
                "Status": "placeholder",
                "Source": source,
                "Notes": "Available while a persisted run is active.",
            },
            {
                "Section": "Pipeline",
                "Metric": "Resume rate",
                "Progress": None,
                "Value": DASHBOARD_PLACEHOLDER,
                "Status": "placeholder",
                "Source": source,
                "Notes": "Tracks how often a run reuses prior stage artifacts.",
            },
        ]

    active_or_completed = [
        stage
        for stage in snapshot.stages.values()
        if stage.progress > 0.0 or stage.status not in {"pending", "placeholder"}
    ]
    resumed_count = sum(bool(stage.resumed) for stage in active_or_completed)
    considered_count = len(active_or_completed)
    resume_rate = 0 if considered_count == 0 else round((resumed_count / considered_count) * 100)

    if snapshot.eta_seconds is None:
        eta_value = "Complete" if snapshot.status.startswith("completed") else "Awaiting estimate"
        eta_status = "live" if snapshot.status.startswith("completed") else "idle"
        eta_notes = (
            "Run has finished all currently scheduled work."
            if snapshot.status.startswith("completed")
            else "ETA appears after the run has enough progress history."
        )
    else:
        eta_value = format_time(snapshot.eta_seconds)
        eta_status = "live"
        eta_notes = "Estimated remaining time for the active run."

    return [
        {
            "Section": "Pipeline",
            "Metric": "Elapsed runtime",
            "Progress": None,
            "Value": format_time(snapshot.elapsed_seconds),
            "Status": "live",
            "Source": source,
            "Notes": "Wall-clock runtime captured in the run snapshot.",
        },
        {
            "Section": "Pipeline",
            "Metric": "ETA",
            "Progress": None,
            "Value": eta_value,
            "Status": eta_status,
            "Source": source,
            "Notes": eta_notes,
        },
        {
            "Section": "Pipeline",
            "Metric": "Resume rate",
            "Progress": None,
            "Value": f"{resume_rate}% ({resumed_count}/{considered_count})",
            "Status": "live",
            "Source": source,
            "Notes": "Proxy for cache/reuse effectiveness across active or completed stages.",
        },
    ]


def _dashboard_breakdown_frame(
    snapshot: Any | None,
    stats: dict[str, Any] | None,
    share_metrics: dict[str, Any],
    *,
    run_dir: str | None = None,
) -> pd.DataFrame:
    """Return a dashboard breakdown table that is safe to render without live data."""
    rows = [
        {
            "Section": "Pipeline",
            "Metric": stage_row["Stage"],
            "Progress": int(stage_row["Progress (%)"]),
            "Value": f"{stage_row['Progress (%)']}%",
            "Status": stage_row["Status"],
            "Source": stage_row["Source"],
            "Notes": stage_row["Detail"],
        }
        for stage_row in _dashboard_stage_frame(snapshot, run_dir=run_dir).to_dict("records")
    ]
    rows.extend(_dashboard_run_throughput_rows(snapshot, run_dir=run_dir))
    rows.extend(_dashboard_optional_task_rows(snapshot, run_dir=run_dir))
    rows.extend(_dashboard_network_stat_rows(stats))
    rows.extend(_dashboard_share_report_rows(share_metrics))
    return pd.DataFrame(rows)


def _dashboard_optional_task_rows(
    snapshot: Any | None, *, run_dir: str | None = None
) -> list[dict[str, Any]]:
    optional_tasks = {} if snapshot is None else snapshot.optional_tasks
    if optional_tasks:
        return [
            {
                "Section": "Optional Tasks",
                "Metric": task_name,
                "Progress": int(task.progress * 100),
                "Value": f"{int(task.progress * 100)}%",
                "Status": task.status,
                "Source": "99_Metadata/run_snapshot.json",
                "Notes": task.detail or "Tracked optional work",
            }
            for task_name, task in sorted(optional_tasks.items())
        ]
    return [
        {
            "Section": "Optional Tasks",
            "Metric": "Tracked tasks",
            "Progress": 0 if snapshot is not None else None,
            "Value": "0 tracked" if snapshot is not None else DASHBOARD_PLACEHOLDER,
            "Status": "idle" if snapshot is not None else "placeholder",
            "Source": (
                "99_Metadata/run_snapshot.json optional_tasks"
                if snapshot is not None
                else _dashboard_snapshot_source(run_dir)
            ),
            "Notes": (
                "No optional tasks have been tracked for this run yet."
                if snapshot is not None
                else "Optional tasks appear after a run snapshot is available."
            ),
        }
    ]


def _dashboard_network_stat_spec(
    stats: dict[str, Any] | None,
) -> list[tuple[str, float | int | None, str]]:
    return [
        ("Strands", None if stats is None else int(stats.get("num_strands", 0)), "count"),
        (
            "Total Length (um)",
            None if stats is None else float(stats.get("total_length", 0.0)),
            "length",
        ),
        (
            "Volume Fraction",
            None if stats is None else float(stats.get("volume_fraction", 0.0)),
            "ratio",
        ),
        (
            "Mean Radius (um)",
            None if stats is None else float(stats.get("mean_radius", 0.0)),
            "radius",
        ),
    ]


def _dashboard_network_value(value: float | int | None, value_type: str) -> tuple[str, str, str]:
    if value is None:
        return (
            DASHBOARD_PLACEHOLDER,
            "placeholder",
            "Loads when a full network result is available in session state",
        )
    if value_type == "count":
        return (f"{value:d}", "live", "Computed from the current network")
    if value_type == "length":
        return (f"{value:.1f}", "live", "Computed from calculate_network_statistics")
    return (
        f"{value:.3f}" if value_type == "ratio" else f"{value:.2f}",
        "live",
        "Computed from calculate_network_statistics",
    )


def _dashboard_network_stat_rows(stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    for metric_name, value, value_type in _dashboard_network_stat_spec(stats):
        display_value, status, notes = _dashboard_network_value(value, value_type)
        rows.append(
            {
                "Section": "Network",
                "Metric": metric_name,
                "Progress": None,
                "Value": display_value,
                "Status": status,
                "Source": "session_state.processing_results",
                "Notes": notes,
            }
        )
    return rows


def _dashboard_share_report_rows(share_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    live_status = "live" if share_metrics else "idle"
    notes_suffix = (
        (
            "Counts report generations in this session",
            "Counts report downloads in this session",
        )
        if share_metrics
        else (
            "No share report generations have been recorded in this session yet",
            "No share report downloads have been recorded in this session yet",
        )
    )
    return [
        {
            "Section": "Share Report",
            "Metric": "Requested",
            "Progress": None,
            "Value": str(share_metrics.get("share_report_requested", 0)),
            "Status": live_status,
            "Source": "session_state.share_report_metrics",
            "Notes": notes_suffix[0],
        },
        {
            "Section": "Share Report",
            "Metric": "Downloaded",
            "Progress": None,
            "Value": str(share_metrics.get("share_report_downloaded", 0)),
            "Status": live_status,
            "Source": "session_state.share_report_metrics",
            "Notes": notes_suffix[1],
        },
    ]


def _build_dashboard_placeholder_trend() -> Any:
    """Build a placeholder trend chart used before network metrics exist."""
    placeholder = pd.DataFrame(
        {
            "Depth Band": ["Surface", "Mid", "Deep"],
            "Coverage": [0.0, 0.0, 0.0],
        }
    )
    fig = px.line(placeholder, x="Depth Band", y="Coverage", markers=True)
    fig.update_traces(line={"dash": "dot", "width": 3}, marker={"size": 9})
    fig.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis_title="Depth band",
        yaxis_title="Coverage",
        showlegend=False,
    )
    return fig


def normalize_dashboard_sections(
    selected_sections,
    *,
    breakdown_sections: tuple[str, ...] = DASHBOARD_BREAKDOWN_SECTIONS,
) -> list[str]:
    """Normalize the selected pills value to a stable list."""
    if not selected_sections:
        return list(breakdown_sections)
    if isinstance(selected_sections, str):
        return [selected_sections]
    return list(selected_sections)


def filter_dashboard_breakdown(
    frame: pd.DataFrame,
    *,
    focus: str = "Overview",
    selected_sections=None,
    show_placeholders: bool = True,
) -> pd.DataFrame:
    """Apply focus and section filters to the breakdown table."""
    filtered = frame.copy()
    if focus == "Network":
        filtered = filtered[filtered["Section"].isin(["Network", "Share Report"])]

    elif focus == "Pipeline":
        filtered = filtered[filtered["Section"].isin(["Pipeline", "Optional Tasks"])]
    if normalized_sections := normalize_dashboard_sections(selected_sections):
        filtered = filtered[filtered["Section"].isin(normalized_sections)]

    if not show_placeholders:
        filtered = filtered[filtered["Status"] != "placeholder"]

    return frame.head(0) if filtered.empty else filtered.reset_index(drop=True)


def build_dashboard_backlog_frame(
    metric_requests: list[dict[str, Any]] | None,
    *,
    repo_url: str,
    release_url: str,
) -> pd.DataFrame:
    """Build an editable backlog for follow-on dashboard work."""
    rows = [
        {
            "Metric": "Run throughput",
            "Owner": "Pipeline",
            "Priority": "High",
            "Tracked": True,
            "Status": "Ready",
            "Reference": repo_url,
            "Notes": "Track runtime, resume rate, and cache effectiveness per run.",
        },
        {
            "Metric": "Dataset inventory",
            "Owner": "Operations",
            "Priority": "Medium",
            "Tracked": False,
            "Status": "TODO",
            "Reference": repo_url,
            "Notes": (
                "Summarize queued, active, and completed volumes. "
                "Requires a persistent run-registry or job-queue entity; "
                "promote to Ready once that data source is defined."
            ),
        },
        {
            "Metric": "Curation QA",
            "Owner": "Analysis",
            "Priority": "Medium",
            "Tracked": False,
            "Status": "TODO",
            "Reference": release_url,
            "Notes": (
                "Reviewer throughput and edge-acceptance-rate metrics. "
                "Requires a structured QA log or curator-session store; "
                "promote to Ready once that data source exists."
            ),
        },
    ]
    rows.extend(
        {
            "Metric": request["metric"],
            "Owner": request["owner"],
            "Priority": request["priority"],
            "Tracked": False,
            "Status": "Requested",
            "Reference": repo_url,
            "Notes": request["notes"] or "Captured from the in-app planning dialog.",
        }
        for request in metric_requests or []
    )
    return pd.DataFrame(rows)
