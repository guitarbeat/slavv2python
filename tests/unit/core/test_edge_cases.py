import numpy as np
import pytest

# Add source path for imports
from slavv.core import SLAVVProcessor
from slavv.core.tracing import (
    _best_watershed_contact_coords,
    _choose_edges_matlab_style,
    _construct_structuring_element_offsets_matlab,
    _finalize_traced_edge,
    _offset_coords_matlab,
    _prune_frontier_indices_beyond_found_vertices,
    _resolve_frontier_edge_connection,
    _supplement_matlab_frontier_candidates_with_watershed_joins,
    _trace_origin_edges_matlab_frontier,
    extract_vertices,
    paint_vertex_center_image,
)


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
    assert vertices["positions"].shape == (0, 3)

    edges = processor.extract_edges(energy_data, vertices, {})
    assert edges["connections"].shape == (0, 2)

    network = processor.construct_network(edges, vertices, {})
    assert len(network["adjacency_list"]) == 0
    assert network["orphans"].size == 0


def test_process_image_requires_3d():
    processor = SLAVVProcessor()
    img2d = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty 3D array"):
        processor.process_image(img2d, {})


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
        "lumen_radius_pixels_axes": np.tile(
            np.array(
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
            (1, 1),
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


def test_finalize_traced_edge_attaches_terminal_at_final_center_hit():
    vertex_positions = np.array([[2, 2, 2], [2, 5, 2]], dtype=np.float32)
    center_image = paint_vertex_center_image(vertex_positions, (8, 8, 8))

    final_trace, metadata = _finalize_traced_edge(
        np.array([[2, 2, 2], [2, 4, 2], [2, 5, 2]], dtype=np.float32),
        stop_reason="max_steps",
        direct_terminal_vertex=None,
        vertex_center_image=center_image,
        vertex_positions=vertex_positions,
        vertex_scales=np.zeros(2, dtype=np.int16),
        lumen_radius_microns=np.ones(2, dtype=np.float32),
        microns_per_voxel=np.ones(3, dtype=np.float32),
        origin_vertex=0,
    )

    assert metadata["terminal_vertex"] == 1
    assert metadata["terminal_resolution"] == "direct_hit"
    assert np.allclose(final_trace[-1], vertex_positions[1])


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


def test_finalize_traced_edge_recovers_terminal_from_near_vertex_fallback():
    vertex_positions = np.array([[2, 2, 2], [2, 5, 2]], dtype=np.float32)
    center_image = paint_vertex_center_image(vertex_positions, (8, 8, 8))

    final_trace, metadata = _finalize_traced_edge(
        np.array([[2, 2, 2], [2, 3, 2], [2, 4, 2]], dtype=np.float32),
        stop_reason="max_steps",
        direct_terminal_vertex=None,
        vertex_center_image=center_image,
        vertex_positions=vertex_positions,
        vertex_scales=np.zeros(2, dtype=np.int16),
        lumen_radius_microns=np.array([1.1, 1.1], dtype=np.float32),
        microns_per_voxel=np.ones(3, dtype=np.float32),
        origin_vertex=0,
    )

    assert metadata["terminal_vertex"] == 1
    assert metadata["terminal_resolution"] == "reverse_near_hit"
    assert np.allclose(final_trace[-1], vertex_positions[1])


def test_finalize_traced_edge_ignores_reentry_into_origin_vertex():
    vertex_positions = np.array([[2, 2, 2], [2, 5, 2]], dtype=np.float32)
    center_image = paint_vertex_center_image(vertex_positions, (8, 8, 8))

    final_trace, metadata = _finalize_traced_edge(
        np.array([[2, 2, 2], [2, 3, 2], [2, 2, 2]], dtype=np.float32),
        stop_reason="bounds",
        direct_terminal_vertex=None,
        vertex_center_image=center_image,
        vertex_positions=vertex_positions,
        vertex_scales=np.zeros(2, dtype=np.int16),
        lumen_radius_microns=np.array([0.5, 0.5], dtype=np.float32),
        microns_per_voxel=np.ones(3, dtype=np.float32),
        origin_vertex=0,
    )

    assert metadata["terminal_vertex"] is None
    assert metadata["terminal_resolution"] is None
    assert len(final_trace) == 3


def test_finalize_traced_edge_keeps_truly_unresolved_trace_dangling():
    vertex_positions = np.array([[2, 2, 2], [2, 6, 2]], dtype=np.float32)
    center_image = paint_vertex_center_image(vertex_positions, (10, 10, 10))

    final_trace, metadata = _finalize_traced_edge(
        np.array([[2, 2, 2], [2, 3, 2], [2, 4, 2]], dtype=np.float32),
        stop_reason="energy_threshold",
        direct_terminal_vertex=None,
        vertex_center_image=center_image,
        vertex_positions=vertex_positions,
        vertex_scales=np.zeros(2, dtype=np.int16),
        lumen_radius_microns=np.array([0.5, 0.5], dtype=np.float32),
        microns_per_voxel=np.ones(3, dtype=np.float32),
        origin_vertex=0,
    )

    assert metadata["terminal_vertex"] is None
    assert metadata["terminal_resolution"] is None
    assert len(final_trace) == 3


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
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert chosen["diagnostics"]["dangling_edge_count"] == 1
    assert chosen["diagnostics"]["duplicate_directed_pair_count"] == 1
    assert chosen["diagnostics"]["antiparallel_pair_count"] == 1


def test_choose_edges_prefers_shorter_equal_energy_duplicate():
    vertex_positions = np.array([[1, 1, 1], [1, 5, 1]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    long_trace = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32)
    short_trace = np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32)
    candidates = {
        "traces": [long_trace, short_trace],
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
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert np.array_equal(chosen["traces"][0], short_trace)
    assert chosen["diagnostics"]["duplicate_directed_pair_count"] == 1


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


def test_offset_coords_matlab_snaps_out_of_bounds_axes_back_to_center():
    offsets = _construct_structuring_element_offsets_matlab(
        np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    coords = _offset_coords_matlab(np.array([0.0, 0.0, 0.0], dtype=np.float32), offsets, (3, 3, 3))

    assert len(coords) == len(offsets)
    assert np.all(coords >= 0)
    assert np.all(coords < 3)
    assert np.any(np.all(coords == np.array([0, 0, 0]), axis=1))


def test_choose_edges_prunes_degree_excess_and_cycles():
    vertex_positions = np.array(
        [[1, 1, 1], [1, 5, 1], [5, 5, 1], [5, 1, 1]],
        dtype=np.float32,
    )
    vertex_scales = np.zeros(4, dtype=np.int16)
    candidates = {
        "traces": [
            np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32),
            np.array([[1, 5, 1], [3, 5, 1], [5, 5, 1]], dtype=np.float32),
            np.array([[5, 5, 1], [3, 3, 1], [1, 1, 1]], dtype=np.float32),
            np.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [1, 2], [2, 0], [0, 3]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0, -3.0, -2.0], dtype=np.float32),
        "energy_traces": [np.array([-5.0, -5.0, -5.0], dtype=np.float32) for _ in range(4)],
        "scale_traces": [np.zeros(3, dtype=np.int16) for _ in range(4)],
        "origin_indices": np.array([0, 1, 2, 0], dtype=np.int32),
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 2},
    )

    assert chosen["connections"].tolist() == [[0, 1], [1, 2]]
    assert chosen["diagnostics"]["degree_pruned_count"] == 1
    assert chosen["diagnostics"]["cycle_pruned_count"] == 1


def test_frontier_tracer_finds_terminal_through_non_monotonic_energy():
    energy = np.full((7, 7, 7), 1.0, dtype=np.float32)
    corridor = {
        (3, 1, 3): -2.0,
        (3, 2, 3): -3.0,
        (3, 3, 3): -1.0,
        (3, 4, 3): -4.0,
        (3, 5, 3): -2.0,
    }
    for coord, value in corridor.items():
        energy[coord] = value

    vertex_positions = np.array([[3.0, 1.0, 3.0], [3.0, 5.0, 3.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 1,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 8.0,
        },
    )

    assert payload["connections"] == [[0, 1]]
    assert np.allclose(payload["traces"][0][-1], vertex_positions[1])
    assert payload["diagnostics"]["stop_reason_counts"]["terminal_frontier_hit"] == 1


def test_frontier_tracer_stops_when_best_frontier_energy_is_nonnegative():
    energy = np.ones((5, 5, 5), dtype=np.float32)
    energy[2, 2, 2] = -2.0
    vertex_positions = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
    vertex_scales = np.array([0], dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 1,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 4.0,
        },
    )

    assert payload["connections"] == []
    assert payload["diagnostics"]["stop_reason_counts"]["frontier_exhausted_nonnegative"] == 1


def test_frontier_tracer_records_length_limited_expansion():
    energy = np.full((7, 7, 7), 1.0, dtype=np.float32)
    energy[3, 1:6, 3] = -5.0
    vertex_positions = np.array([[3.0, 1.0, 3.0], [3.0, 5.0, 3.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 1,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 1.1,
        },
    )

    assert payload["connections"] == []
    assert payload["diagnostics"]["stop_reason_counts"]["length_limit"] > 0


def test_frontier_tracer_matches_matlab_origin_distance_budget():
    energy = np.ones((7, 7, 7), dtype=np.float32)
    energy[3, 1:5, 3] = -5.0
    vertex_positions = np.array([[3.0, 1.0, 3.0], [3.0, 4.0, 3.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 1,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 4.0,
        },
    )

    # MATLAB seeds the origin distance map at 1, so a terminal exactly three
    # voxel steps away is beyond a 4.0 budget and should not be reached.
    assert payload["connections"] == []
    assert payload["diagnostics"]["stop_reason_counts"]["length_limit"] > 0


def test_frontier_tracer_skips_origins_too_close_to_border():
    energy = np.full((5, 5, 5), -2.0, dtype=np.float32)
    vertex_positions = np.array([[0.0, 2.0, 2.0], [4.0, 2.0, 2.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 1,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 8.0,
        },
    )

    assert payload["connections"] == []
    assert payload["diagnostics"]["stop_reason_counts"]["bounds"] == 1


def test_best_watershed_contact_coords_selects_lowest_energy_touch():
    labels = np.zeros((3, 3, 3), dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[1, 0, 0] = 2
    labels[0, 1, 0] = 1
    labels[1, 1, 0] = 2
    energy = np.zeros((3, 3, 3), dtype=np.float32)
    energy[0, 0, 0] = -3.0
    energy[1, 0, 0] = -4.0
    energy[0, 1, 0] = -6.0
    energy[1, 1, 0] = -5.0

    contacts = _best_watershed_contact_coords(labels, energy)

    assert set(contacts) == {(0, 1)}
    assert contacts[(0, 1)].tolist() == [0, 1, 0]


def test_watershed_join_supplement_adds_missing_touching_pair():
    energy = np.full((5, 5, 5), -1.0, dtype=np.float32)
    # Phase 2: provide a frontier candidate so at least one vertex has reachability
    candidates = {
        "traces": [np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)],
        "connections": np.array([[0, -1]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float32),
        "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
        "scale_traces": [np.zeros(2, dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "diagnostics": {},
    }
    vertex_positions = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32)

    supplemented = _supplement_matlab_frontier_candidates_with_watershed_joins(
        candidates,
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        energy_sign=-1.0,
        require_mutual_frontier_participation=False,
    )

    # The supplement should have added the (0, 1) pair
    connections = supplemented["connections"].tolist()
    assert [0, 1] in connections
    assert supplemented["diagnostics"]["watershed_join_supplement_count"] >= 1


def test_watershed_join_supplement_can_require_mutual_frontier_participation():
    energy = np.full((5, 5, 5), -1.0, dtype=np.float32)
    candidates = {
        "traces": [np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)],
        "connections": np.array([[0, -1]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float32),
        "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
        "scale_traces": [np.zeros(2, dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "diagnostics": {},
    }
    vertex_positions = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32)

    supplemented = _supplement_matlab_frontier_candidates_with_watershed_joins(
        candidates,
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        energy_sign=-1.0,
        require_mutual_frontier_participation=True,
    )

    assert supplemented["connections"].tolist() == [[0, -1]]
    assert supplemented["diagnostics"]["watershed_join_supplement_count"] == 0
    assert supplemented["diagnostics"]["watershed_mutual_frontier_rejected"] >= 1


def test_watershed_join_supplement_respects_endpoint_degree_cap():
    energy = np.full((5, 5, 5), -1.0, dtype=np.float32)
    candidates = {
        "traces": [np.array([[0, 0, 0], [0, 4, 0]], dtype=np.float32)],
        "connections": np.array([[0, 2]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float32),
        "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
        "scale_traces": [np.zeros(2, dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "diagnostics": {},
    }
    vertex_positions = np.array(
        [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 0.0]],
        dtype=np.float32,
    )

    supplemented = _supplement_matlab_frontier_candidates_with_watershed_joins(
        candidates,
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        energy_sign=-1.0,
        max_edges_per_vertex=1,
    )

    assert supplemented["connections"].tolist() == [[0, 2]]
    assert supplemented["diagnostics"]["watershed_join_supplement_count"] == 0
    assert supplemented["diagnostics"]["watershed_endpoint_degree_rejected"] >= 1


def test_extract_edges_parity_can_supplement_empty_frontier_candidates(monkeypatch):
    def fake_frontier(*_args, **_kwargs):
        return {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "diagnostics": {},
        }

    monkeypatch.setattr(
        "slavv.core.tracing._generate_edge_candidates_matlab_frontier",
        fake_frontier,
    )

    processor = SLAVVProcessor()
    energy = np.full((5, 5, 5), -1.0, dtype=np.float32)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "energy_sign": -1.0,
        "energy_origin": "matlab_batch_hdf5",
    }
    vertices = {
        "positions": np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32),
        "scales": np.array([0, 0], dtype=np.int16),
    }

    edges = processor.extract_edges(
        energy_data,
        vertices,
        {
            "comparison_exact_network": True,
            "number_of_edges_per_vertex": 1,
            "parity_watershed_candidate_mode": "legacy_supplement",
        },
    )

    # Phase 2+: with empty frontier candidates, reachability gate blocks supplements
    assert edges["connections"].size == 0
    assert edges["diagnostics"].get("watershed_reachability_rejected", 0) >= 1


def test_extract_edges_parity_requires_mutual_frontier_participation(monkeypatch):
    def fake_frontier(*_args, **_kwargs):
        return {
            "traces": [np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)],
            "connections": np.array([[0, -1]], dtype=np.int32),
            "metrics": np.array([-1.0], dtype=np.float32),
            "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
            "scale_traces": [np.zeros(2, dtype=np.int16)],
            "origin_indices": np.array([0], dtype=np.int32),
            "connection_sources": ["frontier"],
            "diagnostics": {},
        }

    monkeypatch.setattr(
        "slavv.core.tracing._generate_edge_candidates_matlab_frontier",
        fake_frontier,
    )

    processor = SLAVVProcessor()
    energy = np.full((5, 5, 5), -1.0, dtype=np.float32)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "energy_sign": -1.0,
        "energy_origin": "matlab_batch_hdf5",
    }
    vertices = {
        "positions": np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32),
        "scales": np.array([0, 0], dtype=np.int16),
    }

    edges = processor.extract_edges(
        energy_data,
        vertices,
        {
            "comparison_exact_network": True,
            "number_of_edges_per_vertex": 1,
            "parity_watershed_candidate_mode": "legacy_supplement",
        },
    )

    assert edges["connections"].size == 0
    assert edges["diagnostics"].get("watershed_mutual_frontier_rejected", 0) >= 1


def test_resolve_frontier_edge_connection_invalidates_better_child_than_parent():
    energy = np.full((5, 5, 5), 1.0, dtype=np.float32)
    shape = energy.shape

    def set_energy(coord, value):
        energy[coord] = value
        return int(coord[0] + coord[1] * shape[0] + coord[2] * shape[0] * shape[1])

    root = set_energy((2, 2, 2), -6.0)
    parent_mid = set_energy((2, 3, 2), -5.0)
    parent_terminal = set_energy((2, 4, 2), -4.0)
    child_terminal = set_energy((3, 2, 2), -7.0)

    current_path = [child_terminal, root]
    parent_path = [parent_terminal, parent_mid, root]
    pointer_index_map = {
        root: -1,
        parent_terminal: -1,
        parent_mid: -1,
    }

    origin_idx, terminal_idx = _resolve_frontier_edge_connection(
        current_path,
        terminal_vertex_idx=2,
        seed_origin_idx=0,
        edge_paths_linear=[parent_path],
        edge_pairs=[(1, 0)],
        pointer_index_map=pointer_index_map,
        energy=energy,
        shape=shape,
    )

    assert origin_idx is None
    assert terminal_idx is None


def test_resolve_frontier_edge_connection_matches_matlab_origin_side_for_root_bifurcation():
    energy = np.full((5, 5, 5), 1.0, dtype=np.float32)
    shape = energy.shape

    def set_energy(coord, value):
        energy[coord] = value
        return int(coord[0] + coord[1] * shape[0] + coord[2] * shape[0] * shape[1])

    parent_origin = set_energy((2, 2, 2), -6.0)
    parent_mid = set_energy((2, 3, 2), -7.0)
    parent_terminal = set_energy((2, 4, 2), -8.0)
    child_terminal = set_energy((3, 2, 2), -5.0)

    current_path = [child_terminal, parent_origin]
    parent_path = [parent_terminal, parent_mid, parent_origin]
    pointer_index_map = {
        parent_origin: -1,
        parent_terminal: -1,
        parent_mid: -1,
    }

    origin_idx, terminal_idx = _resolve_frontier_edge_connection(
        current_path,
        terminal_vertex_idx=2,
        seed_origin_idx=0,
        edge_paths_linear=[parent_path],
        edge_pairs=[(1, 0)],
        pointer_index_map=pointer_index_map,
        energy=energy,
        shape=shape,
    )

    assert origin_idx == 1
    assert terminal_idx == 2


def test_frontier_tracer_does_not_prune_from_invalid_terminal_before_valid_edge(
    monkeypatch,
):
    energy = np.full((5, 5, 5), 1.0, dtype=np.float32)
    energy[2, 2, 2] = -4.0
    energy[2, 3, 2] = -8.0
    energy[3, 2, 2] = -7.0
    energy[3, 3, 2] = -6.0
    vertex_positions = np.array([[2.0, 2.0, 2.0], [2.0, 3.0, 2.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    resolve_calls = {"count": 0}

    def fake_resolve(*_args, **_kwargs):
        resolve_calls["count"] += 1
        return None, None

    def fail_if_pruned(*_args, **_kwargs):
        raise AssertionError("invalid terminal directions should not prune frontier yet")

    monkeypatch.setattr("slavv.core.tracing._resolve_frontier_edge_connection", fake_resolve)
    monkeypatch.setattr(
        "slavv.core.tracing._prune_frontier_indices_beyond_found_vertices",
        fail_if_pruned,
    )

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 1,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 6.0,
        },
    )

    assert resolve_calls["count"] >= 1
    assert payload["connections"] == []


def test_frontier_tracer_invalid_terminal_does_not_consume_edge_budget(monkeypatch):
    energy = np.full((9, 9, 9), 1.0, dtype=np.float32)
    energy[4, 4, 4] = -9.0

    # First branch reaches an invalid terminal, the next two are valid.
    energy[4, 5, 4] = -8.0
    energy[4, 6, 4] = -7.0
    energy[5, 4, 4] = -6.0
    energy[6, 4, 4] = -5.0
    energy[4, 3, 4] = -4.0
    energy[4, 2, 4] = -3.0

    vertex_positions = np.array(
        [
            [4.0, 4.0, 4.0],
            [4.0, 6.0, 4.0],
            [6.0, 4.0, 4.0],
            [4.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )
    vertex_scales = np.zeros(4, dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    resolve_calls = {"count": 0}

    def fake_resolve(*_args, **_kwargs):
        resolve_calls["count"] += 1
        if resolve_calls["count"] == 1:
            return None, None
        if resolve_calls["count"] == 2:
            return 0, 2
        return 0, 3

    monkeypatch.setattr("slavv.core.tracing._resolve_frontier_edge_connection", fake_resolve)

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 2,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 8.0,
        },
    )

    assert resolve_calls["count"] >= 3
    assert payload["connections"] == [[0, 2], [0, 3]]


def test_prune_frontier_indices_beyond_found_vertices_removes_forward_voxels():
    candidates = np.array([[2, 3, 2], [2, 4, 2], [2, 1, 2]], dtype=np.int32)
    pruned = _prune_frontier_indices_beyond_found_vertices(
        candidates,
        origin_position_microns=np.array([2.0, 2.0, 2.0]),
        displacement_vectors=[np.array([0.0, 1.0, 0.0])],
        microns_per_voxel=np.ones(3, dtype=np.float32),
    )

    assert pruned.tolist() == [[2, 3, 2], [2, 1, 2]]
