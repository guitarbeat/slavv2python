from __future__ import annotations

from dev.tests.support.payload_builders import build_energy_result, build_processing_results
from source.apps.analysis_state import (
    build_analysis_connectivity_rows,
    build_analysis_full_stats_rows,
    has_analysis_network,
    normalize_analysis_results,
    resolve_analysis_stats,
)


def test_normalize_analysis_results_returns_plain_dict_payload():
    processing_results = build_processing_results(overrides={"metadata": {"source": "analysis"}})

    normalized = normalize_analysis_results(processing_results)

    assert {"vertices", "edges", "network", "parameters", "energy_data"}.issubset(normalized)
    assert normalized["metadata"] == {"source": "analysis"}


def test_has_analysis_network_requires_network_payload():
    assert has_analysis_network(build_processing_results()) is True
    assert has_analysis_network({"energy_data": build_energy_result()}) is False


def test_resolve_analysis_stats_prefers_existing_stats():
    stats = resolve_analysis_stats(build_processing_results(), {"total_length": 5.0})

    assert stats == {"total_length": 5.0}


def test_resolve_analysis_stats_falls_back_to_processing_counts():
    stats = resolve_analysis_stats(build_processing_results(), None)

    assert stats["Vertices"] == 3
    assert stats["Edges"] == 2


def test_build_analysis_connectivity_rows_uses_expected_metrics():
    rows = build_analysis_connectivity_rows(
        {
            "num_connected_components": 3,
            "avg_path_length": 12.5,
            "clustering_coefficient": 0.42,
            "network_diameter": 18.0,
        }
    )

    assert rows == [
        {"Metric": "Connected Components", "Value": 3},
        {"Metric": "Average Path Length", "Value": 12.5},
        {"Metric": "Clustering Coefficient", "Value": 0.42},
        {"Metric": "Network Diameter", "Value": 18.0},
    ]


def test_build_analysis_full_stats_rows_formats_values_for_display():
    rows = build_analysis_full_stats_rows(
        {
            "num_strands": 7,
            "num_bifurcations": 2,
            "num_endpoints": 4,
            "total_length": 120.25,
            "mean_strand_length": 17.18,
            "length_density": 0.0314,
            "volume_fraction": 0.00456,
            "mean_radius": 2.345,
            "radius_std": 0.678,
            "bifurcation_density": 9.12,
            "surface_area": 45.67,
            "mean_tortuosity": 1.234,
            "num_connected_components": 2,
            "avg_path_length": 6.78,
            "clustering_coefficient": 0.91,
            "network_diameter": 15.55,
            "fractal_dimension": 1.87,
            "lacunarity": 0.44,
            "tortuosity_std": 0.22,
        }
    )

    assert rows[0] == {"Metric": "Number of Strands", "Value": 7}
    assert rows[3] == {"Metric": "Total Length (um)", "Value": "120.2"}
    assert rows[6] == {"Metric": "Volume Fraction", "Value": "0.0046"}
    assert rows[-1] == {"Metric": "Tortuosity Std", "Value": "0.22"}


