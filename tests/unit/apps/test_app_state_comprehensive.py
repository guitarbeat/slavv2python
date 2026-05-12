"""Comprehensive tests for App State management across analysis, dashboard, processing, visualization, and curation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")

from slavv_python.apps.shared_state.dashboard import (
    load_dashboard_context,
    normalize_dashboard_results,
    resolve_dashboard_stats,
)
from slavv_python.apps.shared_state.analysis import (
    build_analysis_connectivity_rows,
    build_analysis_full_stats_rows,
    has_analysis_network,
    normalize_analysis_results,
    resolve_analysis_stats,
)
from slavv_python.apps.shared_state.curation import (
    apply_curated_session_results,
    build_curation_stats_rows,
    summarize_processing_counts,
    sync_curated_processing_results,
)
from slavv_python.apps.shared_state.processing import (
    build_processing_run_dir,
    load_processing_snapshot,
    store_processing_session_state,
    summarize_processing_metrics,
)
from slavv_python.apps.shared_state.visualization import (
    extract_visualization_export_payload,
    has_visualization_network,
    list_available_visualizations,
    normalize_visualization_results,
    resolve_visualization_session_context,
)
from slavv_python.runtime.run_state import RunSnapshot
from tests.support.payload_builders import (
    build_edges_payload,
    build_energy_result,
    build_network_payload,
    build_processing_results,
    build_vertices_payload,
)


def _sample_processing_results():
    vertices = build_vertices_payload(
        positions=[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
        radii_microns=[2.0, 2.5, 3.0],
    )
    edges = build_edges_payload(
        traces=[
            np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
            np.array([[0.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype=float),
        ],
        connections=np.array([[0, 1], [1, 2]], dtype=int),
        energies=np.array([-1.0, -0.8], dtype=float),
    )
    network = build_network_payload(strands=[[0, 1, 2]], bifurcations=np.array([], dtype=int))
    return build_processing_results(
        vertices=vertices,
        edges=edges,
        network=network,
        parameters={"microns_per_voxel": [1.0, 1.0, 1.0], "remove_cycles": True},
    )


# ==============================================================================
# Normalization and Basic State Checks
# ==============================================================================


@pytest.mark.unit
def test_normalization_methods_return_plain_dict_payloads():
    """Test that all state normalization methods return a consistent dictionary structure."""
    res_analysis = normalize_analysis_results(
        build_processing_results(overrides={"metadata": {"slavv_python": "analysis"}})
    )
    res_dashboard = normalize_dashboard_results(
        build_processing_results(overrides={"metadata": {"slavv_python": "dashboard"}})
    )
    res_viz = normalize_visualization_results(
        build_processing_results(overrides={"metadata": {"slavv_python": "viz"}})
    )

    for normalized in [res_analysis, res_dashboard, res_viz]:
        assert {"vertices", "edges", "network", "parameters", "energy_data"}.issubset(normalized)

    assert res_analysis["metadata"] == {"slavv_python": "analysis"}
    assert res_dashboard["metadata"] == {"slavv_python": "dashboard"}
    assert res_viz["metadata"] == {"slavv_python": "viz"}


@pytest.mark.unit
def test_state_network_presence_checks():
    """Test that network presence checks correctly identify full results."""
    full = build_processing_results()
    empty = {"energy_data": build_energy_result()}

    assert has_analysis_network(full) is True
    assert has_analysis_network(empty) is False
    assert has_visualization_network(full) is True
    assert has_visualization_network(empty) is False


# ==============================================================================
# Analysis State Tests
# ==============================================================================


@pytest.mark.unit
def test_resolve_analysis_stats_prefers_existing_stats():
    stats = resolve_analysis_stats(build_processing_results(), {"total_length": 5.0})
    assert stats == {"total_length": 5.0}


@pytest.mark.unit
def test_resolve_analysis_stats_falls_back_to_processing_counts():
    stats = resolve_analysis_stats(build_processing_results(), None)
    assert stats["Vertices"] == 3
    assert stats["Edges"] == 2


@pytest.mark.unit
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


@pytest.mark.unit
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


# ==============================================================================
# Curation State Tests
# ==============================================================================


@pytest.mark.unit
def test_sync_curated_processing_results_rebuilds_network_and_preserves_baseline():
    processing_results = _sample_processing_results()
    curated_vertices = build_vertices_payload(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
        radii_microns=np.array([2.0, 2.5], dtype=float),
    )
    curated_edges = build_edges_payload(
        traces=[np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float)],
        connections=np.array([[0, 1]], dtype=int),
        energies=np.array([-1.0], dtype=float),
    )

    updated_results, baseline_counts, current_counts = sync_curated_processing_results(
        processing_results,
        curated_vertices,
        curated_edges,
    )

    assert baseline_counts == {"Vertices": 3, "Edges": 2, "Strands": 1, "Bifurcations": 0}
    assert current_counts == {"Vertices": 2, "Edges": 1, "Strands": 1, "Bifurcations": 0}
    assert len(updated_results["network"]["strands"]) == 1


@pytest.mark.unit
def test_sync_curated_processing_results_keeps_existing_baseline_counts():
    processing_results = _sample_processing_results()
    baseline_counts = {"Vertices": 10, "Edges": 8, "Strands": 4, "Bifurcations": 1}

    _, preserved_baseline, _ = sync_curated_processing_results(
        processing_results,
        processing_results["vertices"],
        processing_results["edges"],
        baseline_counts=baseline_counts,
    )
    assert preserved_baseline == baseline_counts


@pytest.mark.unit
def test_apply_curated_session_results_updates_session_state():
    processing_results = _sample_processing_results()
    session_state = {
        "processing_results": processing_results,
        "curation_baseline_counts": {"Vertices": 10},
        "share_report_prepared_signature": "signature-123",
    }

    apply_curated_session_results(
        session_state,
        processing_results["vertices"],
        processing_results["edges"],
        curation_mode="Automatic (Rule-based)",
    )

    assert session_state["last_curation_mode"] == "Automatic (Rule-based)"
    assert "share_report_prepared_signature" not in session_state


@pytest.mark.unit
def test_build_curation_stats_rows_reports_signed_change_percentages():
    rows = build_curation_stats_rows(
        {"Vertices": 10, "Edges": 5, "Strands": 2, "Bifurcations": 0},
        {"Vertices": 8, "Edges": 6, "Strands": 3, "Bifurcations": 1},
    )
    assert rows[0] == {
        "Component": "Vertices",
        "Original": 10,
        "Current": 8,
        "Delta": -2,
        "Change (%)": "-20.00",
    }
    assert rows[1]["Change (%)"] == "+20.00"


# ==============================================================================
# Dashboard and Context Tests
# ==============================================================================


@pytest.mark.unit
def test_resolve_dashboard_stats_requires_full_network_results():
    assert resolve_dashboard_stats(None, image_shape=(10, 10, 10)) is None
    assert resolve_dashboard_stats({"parameters": {}}, image_shape=(10, 10, 10)) is None


@pytest.mark.unit
def test_resolve_dashboard_stats_uses_supplied_stats_builder():
    processing_results = build_processing_results()
    stats = resolve_dashboard_stats(
        processing_results,
        image_shape=(10, 10, 10),
        stats_builder=lambda results, *, image_shape: {
            "num_strands": len(results["network"]["strands"]),
            "image_shape": image_shape,
        },
    )
    assert stats == {"num_strands": 1, "image_shape": (10, 10, 10)}


@pytest.mark.unit
def test_load_dashboard_context_normalizes_results_and_stats():
    processing_results = build_processing_results()
    seen_run_dirs: list[str] = []

    context = load_dashboard_context(
        {
            "current_run_dir": "run-dir",
            "processing_results": processing_results,
            "share_report_metrics": {"share_report_requested": 2},
            "dataset_name": "Sample dataset",
            "image_shape": (6, 5, 4),
        },
        snapshot_loader=lambda run_dir: (
            seen_run_dirs.append(run_dir) or RunSnapshot(run_id="run-123", stages={})
        ),
        stats_builder=lambda results, *, image_shape: {
            "num_strands": len(results["network"]["strands"]),
            "shape": image_shape,
        },
    )

    assert seen_run_dirs == ["run-dir"]
    assert context["snapshot"].run_id == "run-123"
    assert context["stats"]["num_strands"] == 1


# ==============================================================================
# Processing State Tests
# ==============================================================================


@pytest.mark.unit
def test_build_processing_run_dir_varies_with_validated_parameters():
    upload_bytes = b"same-uploaded-file"
    first = build_processing_run_dir(upload_bytes, {"radius": 1.5})
    second = build_processing_run_dir(upload_bytes, {"radius": 2.0})
    repeated = build_processing_run_dir(upload_bytes, {"radius": 1.5})
    assert first != second
    assert first == repeated


@pytest.mark.unit
def test_load_processing_snapshot_uses_snapshot_loader():
    seen_run_dirs: list[str] = []
    snapshot = load_processing_snapshot(
        {"current_run_dir": "run-dir"},
        snapshot_loader=lambda run_dir: (
            seen_run_dirs.append(run_dir) or RunSnapshot(run_id="run-123", stages={})
        ),
    )
    assert seen_run_dirs == ["run-dir"]
    assert snapshot.run_id == "run-123"


@pytest.mark.unit
def test_summarize_processing_metrics_counts_normalized_payload_parts():
    metrics = summarize_processing_metrics(build_processing_results())
    assert metrics == {"vertices": 3, "edges": 2, "strands": 1, "bifurcations": 0}


@pytest.mark.unit
def test_store_processing_session_state_persists_processing_outputs():
    session_state = {"curation_baseline_counts": {"vertices": 3}}
    snapshot = RunSnapshot(run_id="run-123", stages={})
    processing_results = build_processing_results(
        overrides={"metadata": {"slavv_python": "processing"}}
    )

    store_processing_session_state(
        session_state,
        results=processing_results,
        validated_params={"radius": 1.5},
        image_shape=(6, 5, 4),
        dataset_name="sample.tif",
        run_dir="run-dir",
        final_snapshot=snapshot,
    )

    assert session_state["current_run_dir"] == "run-dir"
    assert "curation_baseline_counts" not in session_state


# ==============================================================================
# Visualization State Tests
# ==============================================================================


@pytest.mark.unit
def test_list_available_visualizations_includes_network_modes():
    processing_results = build_processing_results()
    available = list_available_visualizations(processing_results)
    assert "3D Network" in available
    assert "Energy Field" in available


@pytest.mark.unit
def test_extract_visualization_export_payload_returns_normalized_components():
    processing_results = build_processing_results()
    v, e, n, p = extract_visualization_export_payload(processing_results)
    assert v["positions"].shape == (3, 3)
    assert e["connections"].shape == (2, 2)
    assert len(n["strands"]) == 1


@pytest.mark.unit
def test_resolve_visualization_session_context_returns_share_defaults():
    context = resolve_visualization_session_context(
        {
            "current_run_dir": "run-dir",
            "dataset_name": "Dataset A",
            "image_shape": (6, 5, 4),
            "share_report_metrics": {"share_report_requested": 2},
        }
    )
    assert context["run_dir"] == "run-dir"
    assert context["share_metrics"] == {"share_report_requested": 2}
