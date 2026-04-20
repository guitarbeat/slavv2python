from __future__ import annotations

from dev.tests.support.payload_builders import build_energy_result, build_processing_results

from slavv.apps.visualization_state import (
    list_available_visualizations,
    normalize_visualization_results,
)


def test_normalize_visualization_results_returns_plain_dict_payload():
    processing_results = build_processing_results(overrides={"metadata": {"source": "viz"}})

    normalized = normalize_visualization_results(processing_results)

    assert {"vertices", "edges", "network", "parameters", "energy_data"}.issubset(normalized)
    assert normalized["metadata"] == {"source": "viz"}
    assert normalized["vertices"]["positions"].shape == (3, 3)


def test_list_available_visualizations_includes_network_modes_for_full_results():
    processing_results = build_processing_results()

    available = list_available_visualizations(processing_results)

    assert available == [
        "Energy Field",
        "2D Network",
        "3D Network",
        "Depth Projection",
        "Strand Analysis",
    ]


def test_list_available_visualizations_handles_energy_only_payload():
    available = list_available_visualizations({"energy_data": build_energy_result()})

    assert available == ["Energy Field"]
