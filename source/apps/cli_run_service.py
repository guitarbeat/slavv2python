"""Run-command helpers for the SLAVV CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def resolve_effective_run_dir(
        *,
        run_dir: str | None,
        output_dir: str,
) -> str | None:
    """Resolve the effective resumable run directory for a CLI run."""
    return run_dir or f"{output_dir}/_slavv_run"


def format_run_event_line(event) -> str:
    """Format a CLI progress event into a single log line."""
    line = (
        f"[{event.stage}] stage={event.stage_progress * 100:.1f}% "
        f"overall={event.overall_progress * 100:.1f}%"
    )
    if event.detail:
        line = f"{line} - {event.detail}"
    return line


def build_exportable_network(results: Mapping[str, Any], *, network_factory):
    """Build a lightweight network object for CLI export commands."""
    vertices = results.get("vertices", {})
    edges = results.get("edges", {})
    pos = np.asarray(vertices.get("positions", []))
    rad = np.asarray(vertices.get("radii_microns", []))
    conn = np.atleast_2d(np.asarray(edges.get("connections", [])))

    return network_factory(
        vertices=pos if pos.size > 0 else np.empty((0, 3)),
        edges=conn if conn.size > 0 else np.empty((0, 2)),
        radii=rad if rad.size > 0 else None,
    )


def filter_export_formats(
        requested_formats: list[str],
        results: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return runnable export formats and any warning messages."""
    warnings: list[str] = []
    export_formats = list(requested_formats)
    if export_formats and "vertices" not in results:
        warnings.append(
            "Export requested but pipeline stopped before extracting vertices. Skipping export."
        )
        return [], warnings
    if export_formats and "network" not in results and "edges" not in results:
        warnings.append(
            "Export requested but pipeline stopped early. Formatting output with available data only."
        )
    return export_formats, warnings


def update_run_export_task(
        *,
        effective_run_dir: str | None,
        export_formats: list[str],
        output_dir: str,
        snapshot_loader,
        context_factory,
        artifact_builder: Callable[[str, list[str]], dict[str, str]],
):
    """Load and update the run snapshot after CLI exports complete."""
    snapshot = None
    if effective_run_dir:
        snapshot = snapshot_loader(effective_run_dir)
        if snapshot is not None:
            context = context_factory(effective_run_dir)
            context.update_optional_task(
                "exports",
                status="completed" if export_formats else "pending",
                detail=(
                    "Exported requested output formats"
                    if export_formats
                    else "No exports requested"
                ),
                artifacts=artifact_builder(output_dir, export_formats),
            )
            snapshot = context.snapshot
    return snapshot


def build_run_completion_lines(
        *,
        effective_run_dir: str | None,
        output_dir: str,
        snapshot,
        status_line_builder,
) -> list[str]:
    """Build the CLI output lines for a completed run."""
    lines: list[str] = []
    if effective_run_dir:
        lines.append(f"Run directory: {effective_run_dir}")
    if snapshot is not None:
        lines.append("")
        lines.extend(status_line_builder(snapshot))
    lines.append(f"Done. Results in {output_dir}")
    return lines


__all__ = [
    "build_exportable_network",
    "build_run_completion_lines",
    "filter_export_formats",
    "format_run_event_line",
    "resolve_effective_run_dir",
    "update_run_export_task",
]
