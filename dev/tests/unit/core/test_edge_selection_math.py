from __future__ import annotations

import numpy as np

from source.core.edges_internal.edge_cleanup import (
    clean_edges_cycles_python,
    clean_edges_orphans_python,
    clean_edges_vertex_degree_excess_python,
)
from source.core.edges_internal.edge_selection_payloads import (
    build_selected_edges_result,
    normalize_candidate_connection_sources,
    prepare_candidate_indices_for_cleanup,
)
from source.core.edges_internal.edge_finalize import prefilter_edge_indices_for_cleanup_matlab_style


def test_normalize_candidate_connection_sources_handles_varied_inputs():
    assert normalize_candidate_connection_sources(["frontier", "WATERSHED"], 2) == [
        "frontier",
        "watershed",
    ]
    assert normalize_candidate_connection_sources(np.array(["geodesic"]), 2) == [
        "geodesic",
        "unknown",
    ]
    assert normalize_candidate_connection_sources(None, 1, default_source="fallback") == ["fallback"]


def test_prepare_candidate_indices_for_cleanup_filters_by_energy_threshold():
    metrics = np.array([-10.0, -5.0, -1.0, 0.0, 5.0], dtype=np.float32)
    indices = prepare_candidate_indices_for_cleanup(
        metrics,
        max_edge_energy=-2.0,
        legacy_route=False,
    )
    assert indices.tolist() == [0, 1]


def test_prepare_candidate_indices_for_cleanup_legacy_route_rejects_nonnegative_candidates():
    metrics = np.array([-10.0, -5.0, -1.0, 0.0, 5.0], dtype=np.float32)
    indices = prepare_candidate_indices_for_cleanup(
        metrics,
        max_edge_energy=10.0,
        legacy_route=True,
    )
    assert indices.tolist() == [0, 1, 2]


def test_clean_edges_cycles_python_removes_cycles_while_preserving_best_edges():
    connections = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int32)
    metrics = np.array([-10.0, -9.0, -8.0, -11.0], dtype=np.float32)
    kept = clean_edges_cycles_python([0, 1, 2, 3], connections, metrics)
    assert set(kept) == {0, 1, 3}


def test_clean_edges_orphans_python_removes_single_edge_components():
    connections = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.int32)
    kept = clean_edges_orphans_python([0, 1, 2], connections)
    assert kept == [1, 2]


def test_clean_edges_vertex_degree_excess_python_limits_degree_by_pruning_worst_edges():
    connections = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    metrics = np.array([-10.0, -5.0, -1.0], dtype=np.float32)
    kept = clean_edges_vertex_degree_excess_python([0, 1, 2], connections, metrics, max_degree=2)
    assert set(kept) == {0, 1}


def test_build_selected_edges_result_aggregates_into_standard_dictionary():
    positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    candidates = {
        "traces": [np.zeros((2, 3))],
        "connections": np.array([[0, 1]]),
        "metrics": np.array([-1.0]),
        "energy_traces": [np.zeros(2)],
        "scale_traces": [np.zeros(2)],
        "connection_sources": ["frontier"],
    }
    result = build_selected_edges_result([0], candidates, positions, {})
    assert result["traces"] == candidates["traces"]
    assert np.array_equal(result["vertex_positions"], positions)
