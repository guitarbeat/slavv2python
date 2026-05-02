from __future__ import annotations

import numpy as np

from source.core._edge_selection.cleanup import (
    clean_edges_cycles_python,
    clean_edges_orphans_python,
    clean_edges_vertex_degree_excess_python,
)
from source.core._edge_selection.payloads import (
    empty_edge_diagnostics,
    prepare_candidate_indices_for_cleanup,
)
from source.core._edges.postprocess import prefilter_edge_indices_for_cleanup_matlab_style


def test_prepare_candidate_indices_for_cleanup_prefers_shorter_mutual_edge_on_tied_metric():
    connections = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.int32,
    )
    metrics = np.array([-5.0, -5.0], dtype=np.float32)
    energy_traces = [
        np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        np.array([-5.0, -5.0], dtype=np.float32),
    ]
    diagnostics = empty_edge_diagnostics()

    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
    )

    assert indices == [1]
    assert diagnostics["antiparallel_pair_count"] == 1


def test_prepare_candidate_indices_for_cleanup_exact_route_keeps_nonnegative_candidates():
    connections = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int32,
    )
    metrics = np.array([-5.0, -4.0], dtype=np.float32)
    energy_traces = [
        np.array([-5.0, -5.0], dtype=np.float32),
        np.array([-4.0, 0.25], dtype=np.float32),
    ]
    diagnostics = empty_edge_diagnostics()

    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
        reject_nonnegative_energy_edges=False,
    )

    assert indices == [0, 1]
    assert diagnostics["negative_energy_rejected_count"] == 0


def test_prepare_candidate_indices_for_cleanup_legacy_route_rejects_nonnegative_candidates():
    connections = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int32,
    )
    metrics = np.array([-5.0, -4.0], dtype=np.float32)
    energy_traces = [
        np.array([-5.0, -5.0], dtype=np.float32),
        np.array([-4.0, 0.25], dtype=np.float32),
    ]
    diagnostics = empty_edge_diagnostics()

    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
    )

    assert indices == [0]
    assert diagnostics["negative_energy_rejected_count"] == 1


def test_clean_edges_vertex_degree_excess_matches_overlapping_vertex_excess_removals():
    connections = np.array(
        [
            [0, 1],
            [2, 0],
            [0, 3],
            [4, 0],
            [4, 5],
            [6, 4],
        ],
        dtype=np.int32,
    )

    keep = clean_edges_vertex_degree_excess_python(
        connections,
        np.zeros((len(connections),), dtype=np.float32),
        max_edges_per_vertex=2,
    )

    assert keep.tolist() == [True, True, False, False, True, False]


def test_cleanup_order_crops_before_degree_cleanup():
    candidate_indices = [0, 1, 2]
    traces = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32),
        np.array([[2.0, 3.0, 2.0], [3.0, 3.0, 2.0]], dtype=np.float32),
    ]
    scale_traces = [np.array([0.0, 0.0], dtype=np.float32) for _ in candidate_indices]
    energy_traces = [
        np.array([-6.0, -6.0], dtype=np.float32),
        np.array([-5.0, -5.0], dtype=np.float32),
        np.array([-4.0, -4.0], dtype=np.float32),
    ]
    connections = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)

    kept_indices, cropped_count = prefilter_edge_indices_for_cleanup_matlab_style(
        candidate_indices,
        traces=traces,
        scale_traces=scale_traces,
        energy_traces=energy_traces,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        size_of_image=(5, 5, 5),
    )
    keep_after_crop = clean_edges_vertex_degree_excess_python(
        connections[kept_indices],
        np.zeros((len(kept_indices),), dtype=np.float32),
        max_edges_per_vertex=2,
    )
    keep_without_crop = clean_edges_vertex_degree_excess_python(
        connections,
        np.zeros((len(connections),), dtype=np.float32),
        max_edges_per_vertex=2,
    )

    assert cropped_count == 1
    assert kept_indices == [1, 2]
    assert keep_after_crop.tolist() == [True, True]
    assert keep_without_crop.tolist() == [True, True, False]


def test_clean_edges_orphans_preserves_edges_that_touch_vertices_or_interiors():
    traces = [
        np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 1.0, 1.0], [3.0, 1.0, 1.0], [4.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 5.0]], dtype=np.float32),
    ]
    vertex_positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
            [4.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    keep = clean_edges_orphans_python(
        traces,
        (8, 8, 8),
        vertex_positions,
    )

    assert keep.tolist() == [True, True, False]


def test_clean_edges_cycles_removes_worst_edge_from_cycle_component_not_first_closer():
    connections = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
        ],
        dtype=np.int32,
    )

    keep = clean_edges_cycles_python(connections)

    assert keep.tolist() == [True, True, True, True, False]
