"""Formatting helpers for CLI-facing reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build_info_lines(version: str, system_info: Mapping[str, object]) -> list[str]:
    """Return the CLI lines for the `slavv info` command."""
    lines = [f"slavv {version}", ""]
    lines.extend(f"  {key}: {value}" for key, value in system_info.items())
    return lines


def build_analysis_output_lines(stats: Mapping[str, Any]) -> list[str]:
    """Return the CLI lines for the `slavv analyze` command."""
    topological_metrics = (
        ("Vertices", stats.get("num_vertices", 0)),
        ("Edges", stats.get("num_edges", 0)),
        ("Strands", stats.get("num_strands", 0)),
        ("Bifurcations", stats.get("num_bifurcations", 0)),
        ("Connected Components", stats.get("num_connected_components", 0)),
        ("Endpoints", stats.get("num_endpoints", 0)),
        ("Mean Degree", stats.get("mean_degree", 0.0)),
        ("Clustering Coefficient", stats.get("clustering_coefficient", 0.0)),
    )
    geometric_metrics = (
        ("Total Edge Length", f"{float(stats.get('total_length', 0.0)):.2f} um"),
        ("Mean Strand Length", f"{float(stats.get('mean_strand_length', 0.0)):.2f} um"),
        ("Mean Edge Length", f"{float(stats.get('mean_edge_length', 0.0)):.2f} um"),
        ("Mean Edge Radius", f"{float(stats.get('mean_edge_radius', 0.0)):.2f} um"),
        ("Mean Radius", f"{float(stats.get('mean_radius', 0.0)):.2f} um"),
        ("Volume Fraction", f"{float(stats.get('volume_fraction', 0.0)):.4f}"),
        ("Bifurcation Density", f"{float(stats.get('bifurcation_density', 0.0)):.2f} /mm^3"),
    )

    lines = ["", "--- Network Statistics ---", "", "Topological Features:"]
    for label, value in topological_metrics:
        if isinstance(value, float):
            lines.append(f"  {label}: {value:.4f}")
        else:
            lines.append(f"  {label}: {value}")

    lines.extend(["", "Geometric Features (Aggregates):"])
    lines.extend(f"  {label}: {value}" for label, value in geometric_metrics)
    return lines


__all__ = ["build_analysis_output_lines", "build_info_lines"]
