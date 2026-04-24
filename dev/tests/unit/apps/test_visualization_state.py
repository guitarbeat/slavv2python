from __future__ import annotations

from dev.tests.support.payload_builders import build_energy_result, build_processing_results
from slavv.apps.visualization_state import (
    extract_visualization_export_payload,
    has_visualization_network,
    list_available_visualizations,
    normalize_visualization_results,
    resolve_visualization_session_context,
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


def test_has_visualization_network_requires_vertices_edges_and_network():
    assert has_visualization_network(build_processing_results()) is True
    assert has_visualization_network({"energy_data": build_energy_result()}) is False


def test_extract_visualization_export_payload_returns_normalized_components():
    processing_results = build_processing_results(overrides={"metadata": {"source": "viz-export"}})

    vertices, edges, network, parameters = extract_visualization_export_payload(processing_results)

    assert vertices["positions"].shape == (3, 3)
    assert edges["connections"].shape == (2, 2)
    assert len(network["strands"]) == 1
    assert parameters == processing_results["parameters"]


def test_resolve_visualization_session_context_returns_share_defaults():
    context = resolve_visualization_session_context(
        {
            "current_run_dir": "run-dir",
            "dataset_name": "Dataset A",
            "image_shape": (6, 5, 4),
            "share_report_metrics": {"share_report_requested": 2},
        }
    )

    assert context == {
        "run_dir": "run-dir",
        "dataset_name": "Dataset A",
        "image_shape": (6, 5, 4),
        "share_metrics": {"share_report_requested": 2},
    }
