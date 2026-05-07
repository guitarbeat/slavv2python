"""Pipeline execution and lifecycle helpers for the SLAVV CLI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...runtime import ProgressEvent, RunSnapshot


def resolve_effective_run_dir(output_dir: str, run_dir_override: str | None) -> str:
    """Choose the structured run directory for metadata tracking."""
    if run_dir_override is not None:
        return run_dir_override
    return os.path.join(output_dir, "_slavv_run")


def format_run_event_line(event: ProgressEvent) -> str:
    """Format a pipeline progress event for CLI printing."""
    return f"[{event.stage}] stage={event.stage_progress * 100:0.1f}% overall={event.overall_progress * 100:0.1f}% - {event.detail}"


def filter_export_formats(requested_formats: list[str]) -> list[str]:
    """Return the set of valid export formats from the user request."""
    if not requested_formats:
        return []
    if "all" in requested_formats:
        return ["json", "mat", "casx", "vmv"]
    return [fmt for fmt in requested_formats if fmt != "csv"]


def update_run_export_task(run_dir: str, artifact_paths: dict[str, str]) -> None:
    """Record export status in the structured run metadata."""
    from ...runtime import RunContext

    run_context = RunContext(run_dir=run_dir, target_stage="network")
    run_context.update_task(
        "exports",
        status="completed",
        progress=1.0,
        detail=f"Generated {len(artifact_paths)} export files",
        artifacts={f"{fmt}_file": os.path.basename(path) for fmt, path in artifact_paths.items()},
    )


def build_run_completion_lines(
    snapshot: RunSnapshot,
    export_artifact_paths: dict[str, str],
) -> list[str]:
    """Format a final summary report for the completed CLI run."""
    lines = [
        "✨ SLAVV Processing Completed Successfully! ✨",
        f"   Run ID:    {snapshot.run_id}",
        f"   Duration:  {snapshot.elapsed_seconds:0.1f}s",
    ]

    if export_artifact_paths:
        lines.append("\nStaged Artifacts:")
        for fmt, path in export_artifact_paths.items():
            lines.append(f"  • {fmt.upper():4} -> {path}")

    return lines


__all__ = [
    "build_run_completion_lines",
    "filter_export_formats",
    "format_run_event_line",
    "resolve_effective_run_dir",
    "update_run_export_task",
]
