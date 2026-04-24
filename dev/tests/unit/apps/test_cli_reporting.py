"""Tests for CLI report formatting helpers."""

from __future__ import annotations

from source.apps.cli_reporting import build_analysis_output_lines, build_info_lines


def test_build_info_lines_includes_version_and_system_info():
    lines = build_info_lines(
        "1.2.3",
        {
            "Python": "3.12.0",
            "Platform": "Windows",
        },
    )

    assert lines == [
        "slavv 1.2.3",
        "",
        "  Python: 3.12.0",
        "  Platform: Windows",
    ]


def test_build_analysis_output_lines_formats_topological_and_geometric_metrics():
    lines = build_analysis_output_lines(
        {
            "num_vertices": 4,
            "num_edges": 3,
            "num_strands": 2,
            "num_bifurcations": 1,
            "num_connected_components": 1,
            "num_endpoints": 2,
            "mean_degree": 1.5,
            "clustering_coefficient": 0.25,
            "total_length": 10.5,
            "mean_strand_length": 5.25,
            "mean_edge_length": 3.5,
            "mean_edge_radius": 1.2,
            "mean_radius": 1.4,
            "volume_fraction": 0.12345,
            "bifurcation_density": 8.2,
        }
    )

    assert lines[:4] == ["", "--- Network Statistics ---", "", "Topological Features:"]
    assert "  Vertices: 4" in lines
    assert "  Mean Degree: 1.5000" in lines
    assert "  Clustering Coefficient: 0.2500" in lines
    assert "Geometric Features (Aggregates):" in lines
    assert "  Total Edge Length: 10.50 um" in lines
    assert "  Volume Fraction: 0.1235" in lines
    assert "  Bifurcation Density: 8.20 /mm^3" in lines


