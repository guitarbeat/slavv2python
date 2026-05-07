"""Helpers for normalized analysis-page state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.apps.state.curation import summarize_processing_counts
from slavv_python.models import normalize_pipeline_result

from . import normalize_state_results

if TYPE_CHECKING:
    from collections.abc import Mapping


def normalize_analysis_results(processing_results: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized dict payload for analysis consumers."""
    return normalize_state_results(processing_results)


def has_analysis_network(processing_results: Mapping[str, Any]) -> bool:
    """Return whether analysis can proceed on the provided results."""
    return normalize_pipeline_result(processing_results).network is not None


def resolve_analysis_stats(
    processing_results: Mapping[str, Any],
    analysis_stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return existing analysis stats or derive baseline counts when absent."""
    if analysis_stats is not None:
        return dict(analysis_stats)
    return summarize_processing_counts(normalize_analysis_results(processing_results))


def build_analysis_connectivity_rows(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the connectivity summary rows shown in the topology panel."""
    return [
        {
            "Metric": "Connected Components",
            "Value": stats.get("num_connected_components", 0),
        },
        {
            "Metric": "Average Path Length",
            "Value": stats.get("avg_path_length", 0.0),
        },
        {
            "Metric": "Clustering Coefficient",
            "Value": stats.get("clustering_coefficient", 0.0),
        },
        {
            "Metric": "Network Diameter",
            "Value": stats.get("network_diameter", 0.0),
        },
    ]


def build_analysis_full_stats_rows(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the full statistics rows shown in the analysis download table."""
    return [
        {"Metric": "Number of Strands", "Value": stats.get("num_strands", 0)},
        {"Metric": "Number of Bifurcations", "Value": stats.get("num_bifurcations", 0)},
        {"Metric": "Number of Endpoints", "Value": stats.get("num_endpoints", 0)},
        {"Metric": "Total Length (um)", "Value": f"{stats.get('total_length', 0):.1f}"},
        {
            "Metric": "Mean Strand Length (um)",
            "Value": f"{stats.get('mean_strand_length', 0):.1f}",
        },
        {"Metric": "Length Density (um/um^3)", "Value": f"{stats.get('length_density', 0):.3f}"},
        {"Metric": "Volume Fraction", "Value": f"{stats.get('volume_fraction', 0):.4f}"},
        {"Metric": "Mean Radius (um)", "Value": f"{stats.get('mean_radius', 0):.2f}"},
        {"Metric": "Radius Std (um)", "Value": f"{stats.get('radius_std', 0):.2f}"},
        {
            "Metric": "Bifurcation Density (/mm^3)",
            "Value": f"{stats.get('bifurcation_density', 0):.2f}",
        },
        {"Metric": "Surface Area (um^2)", "Value": f"{stats.get('surface_area', 0):.1f}"},
        {"Metric": "Mean Tortuosity", "Value": f"{stats.get('mean_tortuosity', 0):.3f}"},
        {
            "Metric": "Number of Connected Components",
            "Value": stats.get("num_connected_components", 0),
        },
        {"Metric": "Average Path Length", "Value": f"{stats.get('avg_path_length', 0):.2f}"},
        {
            "Metric": "Clustering Coefficient",
            "Value": f"{stats.get('clustering_coefficient', 0):.2f}",
        },
        {"Metric": "Network Diameter", "Value": f"{stats.get('network_diameter', 0):.2f}"},
        {"Metric": "Fractal Dimension", "Value": f"{stats.get('fractal_dimension', 0):.2f}"},
        {"Metric": "Lacunarity", "Value": f"{stats.get('lacunarity', 0):.2f}"},
        {"Metric": "Tortuosity Std", "Value": f"{stats.get('tortuosity_std', 0):.2f}"},
    ]


__all__ = [
    "build_analysis_connectivity_rows",
    "build_analysis_full_stats_rows",
    "has_analysis_network",
    "normalize_analysis_results",
    "resolve_analysis_stats",
]
