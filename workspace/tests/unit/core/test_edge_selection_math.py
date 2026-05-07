from __future__ import annotations

import numpy as np
from slavv_python.core.edges_internal.edge_cleanup import (
    clean_edges_cycles_python,
    clean_edges_orphans_python,
    clean_edges_vertex_degree_excess_python,
)
from slavv_python.core.edges_internal.edge_selection_payloads import (
    build_selected_edges_result,
    normalize_candidate_connection_sources,
    prepare_candidate_indices_for_cleanup,
)


def test_normalize_candidate_connection_sources_handles_varied_inputs():
    assert normalize_candidate_connection_sources(["frontier", "WATERSHED"], 2) == [
        "frontier",
        "watershed",
    ]
    assert normalize_candidate_connection_sources(np.array(["geodesic"]), 2) == [
        "geodesic",
        "unknown",
    ]
    assert normalize_candidate_connection_sources(None, 1, default_source="fallback") == [
        "fallback"
    ]


def test_prepare_candidate_indices_for_cleanup_filters_by_energy_threshold():
    connections = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    metrics = np.array([-10.0, -5.0, -1.0, 0.0, 5.0], dtype=np.float32)
    energy_traces = [
        np.array([-10.0]),
        np.array([-5.0]),
        np.array([-1.0]),
        np.array([0.0]),
        np.array([5.0]),
    ]
    diagnostics = {}
    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
        reject_nonnegative_energy_edges=True,
    )
    assert indices == [0, 1, 2]


def test_prepare_candidate_indices_for_cleanup_legacy_route_rejects_nonnegative_candidates():
    connections = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    metrics = np.array([-10.0, -5.0, -1.0, 0.0, 5.0], dtype=np.float32)
    energy_traces = [
        np.array([-10.0]),
        np.array([-5.0]),
        np.array([-1.0]),
        np.array([0.0]),
        np.array([5.0]),
    ]
    diagnostics = {}
    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
        reject_nonnegative_energy_edges=False,
    )
    assert indices == [0, 1, 2, 3, 4]


def test_clean_edges_cycles_python_removes_cycles_while_preserving_best_edges():
    connections = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int32)
    keep_mask = clean_edges_cycles_python(connections)
    assert np.array_equal(keep_mask, np.array([True, True, False, True]))


def test_clean_edges_orphans_python_removes_single_edge_components():
    traces = [
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float32),
    ]
    vertex_positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    keep_mask = clean_edges_orphans_python(traces, (5, 5, 5), vertex_positions)
    assert np.array_equal(keep_mask, np.array([True, False]))


def test_clean_edges_vertex_degree_excess_python_limits_degree_by_pruning_worst_edges():
    connections = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    metrics = np.array([-10.0, -5.0, -1.0], dtype=np.float32)
    keep_mask = clean_edges_vertex_degree_excess_python(connections, metrics, 2)
    assert np.array_equal(keep_mask, np.array([True, True, False]))


def test_build_selected_edges_result_aggregates_into_standard_dictionary():
    positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    traces = [np.zeros((2, 3))]
    connections = np.array([[0, 1]])
    metrics = np.array([-1.0])
    energy_traces = [np.zeros(2)]
    scale_traces = [np.zeros(2)]
    connection_sources = ["frontier"]
    diagnostics = {}
    result = build_selected_edges_result(
        [0],
        traces,
        connections,
        metrics,
        energy_traces,
        scale_traces,
        connection_sources,
        positions,
        diagnostics,
    )
    assert len(result["traces"]) == 1
    assert np.array_equal(result["vertex_positions"], positions)
