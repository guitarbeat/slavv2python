import numpy as np
import pytest

from source.core import SLAVVProcessor
from source.core.edge_primitives import _finalize_traced_edge
from source.core.edge_selection import (
    _choose_edges_matlab_style,
    _construct_structuring_element_offsets_matlab,
    _matlab_edge_endpoint_positions_and_scales,
    _offset_coords_matlab,
    _snapshot_endpoint_influences_matlab,
)
from source.core.edges_internal import edge_selection as conflict_painting_module
from source.core.edges_internal.edge_cleanup import clean_edges_cycles_python
from source.core.graph import _remove_short_hairs
from source.core.vertices import extract_vertices, paint_vertex_center_image


def test_extract_handles_no_vertices():
    processor = SLAVVProcessor()
    energy = np.ones((3, 3, 3), dtype=np.float32)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "energy_sign": -1.0,
    }

    vertices = processor.extract_vertices(energy_data, {})
    edges = processor.extract_edges(energy_data, vertices, {})
    network = processor.construct_network(edges, vertices, {})

    assert vertices["positions"].shape == (0, 3)
    assert edges["connections"].shape == (0, 2)
    assert len(network["adjacency_list"]) == 0
    assert network["orphans"].size == 0


def test_process_image_requires_3d():
    processor = SLAVVProcessor()

    with pytest.raises(ValueError, match="non-empty 3D array"):
        processor.process_image(np.zeros((5, 5), dtype=np.float32), {})


def test_vertex_extraction_uses_matlab_paint_selection():
    energy = np.zeros((12, 12, 12), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    energy[5, 5, 5] = -5.0
    energy[6, 5, 5] = -4.0
    scale_indices[5, 5, 5] = 2
    scale_indices[6, 5, 5] = 2

    energy_data = {
        "energy": energy,
        "scale_indices": scale_indices,
        "lumen_radius_pixels": np.array([0.5, 0.8, 1.0, 1.4, 1.8, 2.2], dtype=np.float32),
        "lumen_radius_pixels_axes": np.array(
            [
                [0.5, 0.5, 0.5],
                [0.8, 0.8, 0.8],
                [1.0, 1.0, 1.0],
                [1.4, 1.4, 1.4],
                [1.8, 1.8, 1.8],
                [2.2, 2.2, 2.2],
            ],
            dtype=np.float32,
        ),
        "lumen_radius_microns": np.array([0.5, 0.8, 1.0, 1.4, 1.8, 2.2], dtype=np.float32),
        "energy_sign": -1.0,
    }
    params = {
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "length_dilation_ratio": 1.0,
    }

    vertices = extract_vertices(energy_data, params)

    assert len(vertices["positions"]) == 1
    assert np.allclose(vertices["positions"][0], [5, 5, 5])


def test_finalize_traced_edge_recovers_reverse_center_hit_on_long_trace():
    vertex_positions = np.array([[2, 2, 2], [2, 5, 2]], dtype=np.float32)
    center_image = paint_vertex_center_image(vertex_positions, (9, 9, 9))

    final_trace, metadata = _finalize_traced_edge(
        np.array([[2, 2, 2], [2, 4, 2], [2, 5, 2], [2, 6, 2], [2, 7, 2]], dtype=np.float32),
        stop_reason="bounds",
        direct_terminal_vertex=None,
        vertex_center_image=center_image,
        vertex_positions=vertex_positions,
        vertex_scales=np.zeros(2, dtype=np.int16),
        lumen_radius_microns=np.ones(2, dtype=np.float32),
        microns_per_voxel=np.ones(3, dtype=np.float32),
        origin_vertex=0,
    )

    assert metadata["terminal_vertex"] == 1
    assert metadata["terminal_resolution"] == "reverse_center_hit"
    assert np.allclose(final_trace[-1], vertex_positions[1])


def test_choose_edges_filters_nonterminal_duplicates_and_antiparallel():
    vertex_positions = np.array([[1, 1, 1], [1, 5, 1]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    candidates = {
        "traces": [
            np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32),
            np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]], dtype=np.float32),
            np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32),
            np.array([[1, 5, 1], [1, 3, 1], [1, 1, 1]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [0, -1], [0, 1], [1, 0]], dtype=np.int32),
        "metrics": np.array([-5.0, -6.0, -4.0, -3.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-6.0, -6.0, -6.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0], dtype=np.float32),
            np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16) for _ in range(4)],
        "origin_indices": np.array([0, 0, 0, 1], dtype=np.int32),
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert chosen["diagnostics"]["dangling_edge_count"] == 1
    assert chosen["diagnostics"]["duplicate_directed_pair_count"] == 1
    assert chosen["diagnostics"]["antiparallel_pair_count"] == 1


def test_choose_edges_tracks_conflict_provenance_by_source():
    vertex_positions = np.array(
        [[1, 1, 1], [1, 5, 1], [1, 3, 1], [3, 5, 1]],
        dtype=np.float32,
    )
    vertex_scales = np.zeros(4, dtype=np.int16)
    chosen_frontier = np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32)
    rejected_watershed = np.array(
        [[1, 3, 1], [1, 4, 1], [1, 5, 1], [2, 5, 1], [3, 5, 1]],
        dtype=np.float32,
    )
    candidates = {
        "traces": [chosen_frontier, rejected_watershed],
        "connections": np.array([[0, 1], [2, 3]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16), np.zeros(5, dtype=np.int16)],
        "origin_indices": np.array([0, 2], dtype=np.int32),
        "connection_sources": ["frontier", "watershed"],
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert chosen["connection_sources"] == ["frontier"]
    assert chosen["diagnostics"]["conflict_rejected_count"] == 1
    assert chosen["diagnostics"]["conflict_rejected_by_source"] == {"watershed": 1}
    assert chosen["diagnostics"]["conflict_blocking_source_counts"] == {"frontier": 1}
    assert chosen["diagnostics"]["conflict_source_pairs"] == {"watershed->frontier": 1}


def test_choose_edges_prefers_shorter_duplicate_pair_when_metrics_are_equal():
    vertex_positions = np.array([[1, 1, 1], [1, 6, 1]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    first_trace = np.array([[1, 1, 1], [1, 2, 1], [1, 4, 1], [1, 6, 1]], dtype=np.float32)
    shorter_duplicate = np.array([[1, 1, 1], [1, 4, 1], [1, 6, 1]], dtype=np.float32)
    candidates = {
        "traces": [first_trace, shorter_duplicate],
        "connections": np.array([[0, 1], [0, 1]], dtype=np.int32),
        "metrics": np.array([-5.0, -5.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(4, dtype=np.int16), np.zeros(3, dtype=np.int16)],
        "origin_indices": np.array([0, 0], dtype=np.int32),
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["chosen_candidate_indices"].tolist() == [1]
    assert np.array_equal(chosen["traces"][0], shorter_duplicate)
    assert chosen["diagnostics"]["duplicate_directed_pair_count"] == 1


def test_choose_edges_exact_route_uses_seeded_trace_permutation(monkeypatch):
    seed_calls: list[int] = []
    permutation_calls: list[int] = []

    class FakeRng:
        def permutation(self, length: int) -> np.ndarray:
            permutation_calls.append(length)
            return np.arange(length - 1, -1, -1, dtype=np.int64)

    def fake_default_rng(seed: int) -> FakeRng:
        seed_calls.append(seed)
        return FakeRng()

    monkeypatch.setattr(conflict_painting_module.np.random, "default_rng", fake_default_rng)

    vertex_positions = np.array([[1, 1, 1], [1, 5, 1]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    candidates = {
        "traces": [np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32)],
        "connections": np.array([[0, 1]], dtype=np.int32),
        "metrics": np.array([-5.0], dtype=np.float32),
        "energy_traces": [np.array([-5.0, -5.0, -5.0], dtype=np.float32)],
        "scale_traces": [np.zeros(3, dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "matlab_global_watershed_exact": True,
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4, "comparison_exact_network": True},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert seed_calls == [conflict_painting_module.EXACT_ROUTE_CHOOSER_SEED]
    assert permutation_calls == [3]
    assert chosen["diagnostics"]["exact_route_chooser_seed"] == (
        conflict_painting_module.EXACT_ROUTE_CHOOSER_SEED
    )


def test_cycle_cleanup_removes_worst_edge_per_cycle_component():
    connections = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [3, 4],
            [4, 5],
            [5, 3],
        ],
        dtype=np.int32,
    )

    keep = clean_edges_cycles_python(connections)

    assert keep.tolist() == [True, True, False, True, True, False]


def test_remove_short_hairs_repeats_until_graph_is_stable():
    adjacency_list = {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2},
    }
    graph_edges = {
        (0, 1): np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        (1, 2): np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        (2, 3): np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float32),
    }

    _remove_short_hairs(graph_edges, adjacency_list, np.ones(3, dtype=np.float32), 1.5)

    assert not graph_edges
    assert adjacency_list == {0: set(), 1: set(), 2: set(), 3: set()}


def test_offset_coords_matlab_snaps_out_of_bounds_axes_back_to_center():
    offsets = _construct_structuring_element_offsets_matlab(
        np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    coords = _offset_coords_matlab(np.array([0.0, 0.0, 0.0], dtype=np.float32), offsets, (3, 3, 3))

    assert len(coords) == len(offsets)
    assert np.all(coords >= 0)
    assert np.all(coords < 3)
    assert np.any(np.all(coords == np.array([0, 0, 0]), axis=1))


def test_matlab_edge_endpoint_positions_and_scales_use_trace_endpoints():
    trace = np.array(
        [[5.2, 5.1, 4.9], [5.0, 6.0, 5.0], [4.8, 7.2, 5.1]],
        dtype=np.float32,
    )
    scale_trace = np.array([1, 2, 3], dtype=np.int16)

    start, end = _matlab_edge_endpoint_positions_and_scales(trace, scale_trace)

    assert np.allclose(start[0], trace[0])
    assert np.allclose(end[0], trace[-1])
    assert start[1] == 1
    assert end[1] == 3


def test_snapshot_endpoint_influences_matlab_restores_overlap_from_combined_snapshot():
    painted_image = np.zeros((5, 5, 5), dtype=np.int32)
    painted_source_image = np.zeros((5, 5, 5), dtype=np.uint8)
    painted_image[1, 1, 1] = 5
    painted_image[1, 2, 1] = 7
    painted_image[1, 3, 1] = 9
    painted_source_image[1, 1, 1] = 1
    painted_source_image[1, 2, 1] = 2
    painted_source_image[1, 3, 1] = 3

    combined_coords, snapshot, source_snapshot = _snapshot_endpoint_influences_matlab(
        [
            np.array([[1, 1, 1], [1, 2, 1]], dtype=np.int32),
            np.array([[1, 2, 1], [1, 3, 1]], dtype=np.int32),
        ],
        painted_image,
        painted_source_image,
    )

    assert np.all(
        painted_image[combined_coords[:, 0], combined_coords[:, 1], combined_coords[:, 2]] == 0
    )
    assert np.all(
        painted_source_image[
            combined_coords[:, 0],
            combined_coords[:, 1],
            combined_coords[:, 2],
        ]
        == 0
    )

    painted_image[combined_coords[:, 0], combined_coords[:, 1], combined_coords[:, 2]] = snapshot
    painted_source_image[
        combined_coords[:, 0],
        combined_coords[:, 1],
        combined_coords[:, 2],
    ] = source_snapshot

    assert painted_image[1, 2, 1] == 7
    assert painted_source_image[1, 2, 1] == 2


def test_choose_edges_uses_trace_endpoint_scales_for_vertex_influence():
    vertex_positions = np.array(
        [[5, 5, 5], [5, 7, 5], [5, 9, 5], [5, 11, 5]],
        dtype=np.float32,
    )
    vertex_scales = np.array([3, 3, 3, 3], dtype=np.int16)
    candidates = {
        "traces": [
            np.array([[5, 5, 5], [5, 6, 5], [5, 7, 5]], dtype=np.float32),
            np.array([[5, 9, 5], [5, 10, 5], [5, 11, 5]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [2, 3]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16), np.zeros(3, dtype=np.int16)],
        "origin_indices": np.array([0, 2], dtype=np.int32),
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.49, 1.0, 1.5, 2.5], dtype=np.float32),
        np.array(
            [
                [0.49, 0.49, 0.49],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5],
                [2.5, 2.5, 2.5],
            ],
            dtype=np.float32,
        ),
        (20, 20, 20),
        {
            "number_of_edges_per_vertex": 4,
            "sigma_per_influence_vertices": 1.0,
            "sigma_per_influence_edges": 0.5,
        },
    )

    assert chosen["connections"].tolist() == [[0, 1], [2, 3]]
