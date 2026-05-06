"""Formatting helpers for CLI-facing reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build_info_lines(version: str, system_info: Mapping[str, object]) -> list[str]:
    """Build the printable lines for the CLI info command."""
    lines = [
        "SLAVV - Segmentation-Less, Automated, Vascular Vectorization",
        f"Version: {version}",
        "",
        "System Information:",
    ]

    for key, value in system_info.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label:15} {value}")

    return lines


def build_analysis_output_lines(
    results: Mapping[str, Any],
    aggregate_metrics: list[tuple[str, object]],
    geometric_metrics: list[tuple[str, object]],
) -> list[str]:
    """Format the results of a network analysis for CLI printing."""
    lines = [
        "Aggregate Network Metrics:",
    ]

    for label, value in aggregate_metrics:
        lines.append(f"  {label:30} {value}")

    lines.extend(["", "Geometric Features (Aggregates):"])
    lines.extend(f"  {label}: {value}" for label, value in geometric_metrics)
    return lines


__all__ = ["build_analysis_output_lines", "build_info_lines"]
