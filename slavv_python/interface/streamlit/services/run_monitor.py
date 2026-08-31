"""Streamlit-facing run operations panel backed by the shared monitor service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from slavv_python.engine.constants import PIPELINE_STAGES
from slavv_python.engine.state import RunSnapshot
from slavv_python.interface.cli.monitor_service import (
    EnergyProgress,
    compute_energy_progress,
    format_duration,
    format_energy_progress_line,
    live_overall_progress,
    load_run_monitor_view,
    status_style,
    tail_log_lines,
)

_STATUS_BADGE_COLORS = {
    "green": "green",
    "red": "red",
    "yellow": "orange",
    "cyan": "blue",
    "grey62": "gray",
    "white": "gray",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def infer_pipeline_route(run_dir: str | Path, snapshot: RunSnapshot | None = None) -> str:
    """Return a user-facing Paper Path vs Exact Route label for a run directory."""
    root = Path(run_dir).expanduser()
    params = _read_json(root / "99_Metadata" / "validated_params.json") or {}
    edge_method = str(params.get("edge_method") or "").casefold()
    if edge_method == "watershed":
        return "Exact Route"
    if (root / "99_Metadata" / "parity_job.json").is_file():
        return "Exact Route (parity job)"
    if edge_method == "tracing":
        profile = str(params.get("pipeline_profile") or "paper")
        if profile == "paper":
            return "Paper Path"
        return "Paper Path (MATLAB-compat defaults)"
    provenance = (snapshot.provenance if snapshot else {}) or {}
    if str(provenance.get("slavv_python") or "").casefold() == "pipeline":
        return "Paper Path"
    return "Pipeline route unknown"


def _iso_age_seconds(value: str | None, now: datetime | None = None) -> float | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (reference - timestamp).total_seconds())


def format_age(seconds: float | None) -> str:
    """Human-readable heartbeat / update age."""
    if seconds is None:
        return "unknown"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def heartbeat_age_seconds(run_dir: str | Path, snapshot: RunSnapshot | None) -> float | None:
    """Return the freshest heartbeat age across resume state and snapshot clocks."""
    root = Path(run_dir).expanduser()
    resume = _read_json(root / "02_Energy" / "resume_state.json") or {}
    ages = [
        _iso_age_seconds(resume.get("heartbeat_at")),
        _iso_age_seconds(snapshot.updated_at if snapshot else None),
    ]
    known = [age for age in ages if age is not None]
    return min(known) if known else None


def load_run_ops_payload(run_dir: str | Path, *, max_log_lines: int = 5) -> dict[str, Any]:
    """Build a plain-data monitor payload for Streamlit (no widget side effects)."""
    view = load_run_monitor_view(run_dir)
    energy = compute_energy_progress(view)
    log_name, log_lines = tail_log_lines(view, max_lines=max_log_lines)
    return {
        "effective_status": view.effective_status,
        "status_reason": view.status_reason,
        "log_paths": tuple(str(path) for path in view.log_paths),
        "log_tail": {"name": log_name, "lines": tuple(log_lines)},
        "energy": None
        if energy is None
        else {
            "units_total": energy.units_total,
            "durable_units_completed": energy.durable_units_completed,
            "live_units_completed": energy.live_units_completed,
            "chunks_in_batch": energy.chunks_in_batch,
            "per_chunk_seconds": energy.per_chunk_seconds,
            "fraction": energy.fraction,
            "eta_seconds": energy.eta_seconds,
            "is_live": energy.is_live,
            "summary": format_energy_progress_line(energy),
        },
    }


@st.cache_data(ttl=5, show_spinner=False)
def cached_run_monitor_view(run_dir: str) -> dict[str, Any]:
    """Cache the monitor view as plain data for Streamlit reruns."""
    return load_run_ops_payload(run_dir)


def _status_badge_color(status: str) -> str:
    rich_color = status_style(status)
    return _STATUS_BADGE_COLORS.get(rich_color, "gray")


def build_stage_unit_rows(snapshot: RunSnapshot | None) -> list[dict[str, Any]]:
    """Return per-stage unit progress rows for dashboard / run-ops bars."""
    if snapshot is None:
        return []
    rows: list[dict[str, Any]] = []
    for stage_name in PIPELINE_STAGES:
        stage = snapshot.stages.get(stage_name)
        if stage is None:
            continue
        total = max(0, int(stage.units_total or 0))
        completed = max(0, int(stage.units_completed or 0))
        if total > 0:
            completed = min(completed, total)
            fraction = completed / total
            units_label = f"{completed}/{total}"
        else:
            fraction = max(0.0, min(1.0, float(stage.progress)))
            units_label = "—"
        rows.append(
            {
                "stage": stage_name,
                "status": stage.status,
                "units_completed": completed,
                "units_total": total,
                "fraction": fraction,
                "units_label": units_label,
            }
        )
    return rows


def render_stage_unit_bars(snapshot: RunSnapshot | None) -> None:
    """Render compact progress bars for stages with known unit cursors."""
    rows = [row for row in build_stage_unit_rows(snapshot) if row["units_total"] > 0]
    if not rows:
        return
    st.caption("Stage units")
    for row in rows:
        st.progress(
            row["fraction"],
            text=(
                f"{row['stage'].title()}: {row['units_label']} "
                f"({row['fraction'] * 100:.0f}%) · {row['status']}"
            ),
        )


def render_run_ops_panel(
    run_dir: str | Path,
    *,
    snapshot: RunSnapshot | None = None,
    expanded: bool = True,
    show_log_hint: bool = True,
) -> None:
    """Render a compact run-operations panel for parity and in-app runs."""
    resolved = str(Path(run_dir).expanduser().resolve())
    cached = cached_run_monitor_view(resolved)
    energy_data = cached.get("energy")
    energy: EnergyProgress | None
    if energy_data is None:
        energy = None
    else:
        energy = EnergyProgress(
            units_total=int(energy_data["units_total"]),
            durable_units_completed=int(energy_data.get("durable_units_completed", 0)),
            live_units_completed=int(energy_data["live_units_completed"]),
            chunks_in_batch=int(energy_data.get("chunks_in_batch", 0)),
            per_chunk_seconds=energy_data.get("per_chunk_seconds"),
            eta_seconds=energy_data.get("eta_seconds"),
            is_live=bool(energy_data.get("is_live")),
            log_path=None,
        )
    route = infer_pipeline_route(resolved, snapshot)

    with st.expander("Run operations", expanded=expanded, icon=":material/monitor_heart:"):
        header, badge = st.columns([3, 1], vertical_alignment="center")
        with header:
            st.markdown(f"**{route}** · `{Path(resolved).name}`")
        with badge:
            st.badge(
                cached["effective_status"].replace("-", " ").title(),
                color=_status_badge_color(cached["effective_status"]),
            )

        heartbeat = heartbeat_age_seconds(resolved, snapshot)
        metrics = st.columns(4, gap="small")
        metrics[0].metric("Current stage", (snapshot.current_stage if snapshot else "") or "idle")
        metrics[1].metric("Heartbeat", format_age(heartbeat))
        if snapshot is not None and snapshot.eta_seconds is not None:
            metrics[2].metric("Stage ETA", format_duration(float(snapshot.eta_seconds)))
        elif energy is not None and energy.eta_seconds is not None:
            metrics[2].metric("Chunk ETA", format_duration(energy.eta_seconds))
        else:
            metrics[2].metric("ETA", "—")
        if snapshot is not None:
            overall = (
                live_overall_progress(snapshot, energy)
                if energy is not None
                else float(snapshot.overall_progress)
            )
            metrics[3].metric("Overall", f"{overall * 100:.1f}%")
        else:
            metrics[3].metric("Overall", "—")

        if snapshot is not None and snapshot.current_detail:
            st.caption(snapshot.current_detail)
        if snapshot is not None:
            render_stage_unit_bars(snapshot)
        if energy_data is not None:
            st.caption(str(energy_data["summary"]))
        elif snapshot is not None:
            energy_stage = snapshot.stages.get("energy")
            if energy_stage is not None and energy_stage.units_total:
                st.caption(
                    f"Energy merge cursor: {energy_stage.units_completed}/"
                    f"{energy_stage.units_total} "
                    f"({energy_stage.progress * 100:.1f}%) — lags parallel compute under n_jobs>1"
                )

        st.caption(cached["status_reason"])
        log_tail = cached.get("log_tail") or {}
        log_lines = list(log_tail.get("lines") or ())
        if log_lines:
            log_label = log_tail.get("name") or "run log"
            st.caption(f"Latest log ({log_label})")
            st.code("\n".join(log_lines), language=None)
        if show_log_hint and cached["log_paths"]:
            st.caption(
                "Live chunk rate: `uv run python scripts/monitor/throughput.py "
                f"--run-dir {resolved} --log <run-log> --total-chunks <N>`"
            )


def render_sidebar_run_pulse(run_dir: str | Path, snapshot: RunSnapshot | None = None) -> None:
    """Render a one-line sidebar pulse for the active run directory."""
    resolved = str(Path(run_dir).expanduser().resolve())
    cached = cached_run_monitor_view(resolved)
    route = infer_pipeline_route(resolved, snapshot)
    heartbeat = heartbeat_age_seconds(resolved, snapshot)
    stage = (snapshot.current_stage if snapshot else "") or "idle"
    st.sidebar.caption(
        f"{route} · {stage.title()} · heartbeat {format_age(heartbeat)} · "
        f"{cached['effective_status'].replace('-', ' ')}"
    )


__all__ = [
    "build_stage_unit_rows",
    "cached_run_monitor_view",
    "format_age",
    "heartbeat_age_seconds",
    "infer_pipeline_route",
    "load_run_ops_payload",
    "render_run_ops_panel",
    "render_sidebar_run_pulse",
    "render_stage_unit_bars",
]
