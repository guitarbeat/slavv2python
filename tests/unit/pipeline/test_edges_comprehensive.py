"""Comprehensive tests for edge extraction, selection, and post-processing.
Consolidated from multiple edge-related test files.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from slavv_python.engine import SlavvPipeline
from slavv_python.pipeline.edges import selection as conflict_painting_module
from slavv_python.pipeline.edges.candidate_manifest import (
    CandidateManifest,
    _append_candidate_unit,
)
from slavv_python.pipeline.edges.cleanup import (
    break_graph_cycles,
    remove_excess_vertex_degrees,
)
from slavv_python.pipeline.edges.discovery import (
    FrontierTracingDiscovery,
    MaintainedTracingDiscovery,
    TracingDiscovery,
    WatershedDiscovery,
    _use_matlab_frontier_tracer,
    _use_watershed_discovery,
)
from slavv_python.pipeline.edges.finalize import (
    _matlab_crop_edges_v200,
    _matlab_edge_endpoint_energy,
    normalize_edges_matlab_style,
    prefilter_edge_indices_for_cleanup_matlab_style,
)
from slavv_python.pipeline.edges.primitives import _finalize_traced_edge
from slavv_python.pipeline.edges.selection import (
    _choose_edges_matlab_style,
    _construct_structuring_element_offsets_matlab,
    _matlab_edge_endpoint_positions_and_scales,
    _offset_coords_matlab,
    _snapshot_endpoint_influences_matlab,
)
from slavv_python.pipeline.edges.selection_payloads import (
    normalize_candidate_connection_sources,
    prepare_candidate_indices_for_cleanup,
)
from slavv_python.pipeline.network import _remove_short_hairs
from slavv_python.pipeline.vertices import (
    extract_vertices,
    paint_vertex_center_image,
)
from slavv_python.schema.results import EnergyResult, VertexSet

# ==============================================================================
# Edge Cases and Lifecycle
# ==============================================================================


@pytest.mark.unit
def test_extract_handles_no_vertices():
    processor = SlavvPipeline()
    energy = np.ones((3, 3, 3), dtype=np.float32)
    energy_data = EnergyResult.from_dict(
        {
            "energy": energy,
            "scale_indices": np.zeros_like(energy, dtype=np.int16),
            "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
            "lumen_radius_microns": np.array([1.0], dtype=np.float32),
            "energy_sign": -1.0,
        }
    )

    vertices = processor.extract_vertices(energy_data, {})
    edges = processor.extract_edges(energy_data, vertices, {})
    network = processor.build_network(edges, vertices, {})

    assert vertices.positions.shape == (0, 3)
    assert edges.connections.shape == (0, 2)
    assert len(network.extra["adjacency_list"]) == 0


@pytest.mark.unit
def test_process_image_requires_3d():
    processor = SlavvPipeline()
    with pytest.raises(ValueError, match="non-empty 3D array"):
        processor.run(np.zeros((5, 5), dtype=np.float32), {})


# ==============================================================================
# Direction Seeding and Tracing (Consolidated from test_edge_direction_seeding.py)
# ==============================================================================


@pytest.mark.unit
@patch(
    "slavv_python.pipeline.edges.trace_directions.estimate_vessel_directions",
    return_value=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float),
)
def test_extract_edges_seeds_directions_with_hessian(mock_generate_directions):
    processor = SlavvPipeline()
    size = 21
    coords = np.indices((size, size, size))
    x, _, z = coords[0] - size // 2, coords[1] - size // 2, coords[2] - size // 2
    energy = -(x**2 + z**2).astype(float)

    energy_data = EnergyResult.from_dict(
        {
            "energy": energy,
            "scale_indices": np.zeros_like(energy, dtype=np.int16),
            "lumen_radius_pixels": np.array([2.0], dtype=float),
            "lumen_radius_microns": np.array([2.0], dtype=float),
            "lumen_radius_pixels_axes": np.array([[2.0, 2.0, 2.0]], dtype=float),
            "energy_sign": -1.0,
        }
    )
    vertices = VertexSet.from_dict(
        {
            "positions": np.array([[10.0, 10.0, 10.0]], dtype=float),
            "scales": np.array([0], dtype=int),
        }
    )
    params = {
        "number_of_edges_per_vertex": 2,
        "step_size_per_origin_radius": 2.0,
        "length_dilation_ratio": 5.0,
        "microns_per_voxel": [1.0, 1.0, 1.0],
    }
    edges = processor.extract_edges(energy_data, vertices, params)
    assert edges.extra["diagnostics"]["candidate_traced_edge_count"] == 2


@pytest.mark.unit
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


# ==============================================================================
# Edge Selection Math (Consolidated from test_edge_selection_math.py)
# ==============================================================================


@pytest.mark.unit
def test_normalize_candidate_connection_sources_handles_varied_inputs():
    assert normalize_candidate_connection_sources(["frontier", "WATERSHED"], 2) == [
        "frontier",
        "watershed",
    ]


@pytest.mark.unit
def test_prepare_candidate_indices_for_cleanup_filters_by_energy_threshold():
    connections = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    metrics = np.array([-10.0, -5.0, 5.0], dtype=np.float32)
    energy_traces = [np.array([-10.0]), np.array([-5.0]), np.array([5.0])]
    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        {},
        reject_nonnegative_energy_edges=True,
    )
    assert indices == [0, 1]


@pytest.mark.unit
def test_prepare_candidate_indices_for_cleanup_uses_float64_energy_metric_for_sorting():
    connections = np.array([[0, 1], [2, 3]], dtype=np.int32)
    metrics = np.array([-13.758037567138672, -13.758037567138672], dtype=np.float32)
    energy_traces = [
        np.array([-20.0, -13.758037661352596] * 4, dtype=np.float64)[:7],
        np.array([-20.0, -13.758037107868688] * 3, dtype=np.float64),
    ]

    indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        {},
        reject_nonnegative_energy_edges=False,
    )

    assert indices == [0, 1]


@pytest.mark.unit
def test_remove_excess_vertex_degrees_limits_degree_by_pruning_worst_edges():
    connections = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    metrics = np.array([-10.0, -5.0, -1.0], dtype=np.float32)
    keep_mask = remove_excess_vertex_degrees(connections, metrics, 2)
    assert np.array_equal(keep_mask, np.array([True, True, False]))


# ==============================================================================
# Post-processing and Finalization (Consolidated from test_edge_postprocess_math.py)
# ==============================================================================


@pytest.mark.unit
def test_matlab_edge_endpoint_energy_matches_geometric_mean_formula():
    trace = np.array([-4.0, -9.0], dtype=np.float32)
    assert _matlab_edge_endpoint_energy(trace) == np.float32(-6.0)


@pytest.mark.unit
def test_normalize_edges_matlab_style_matches_vectorize_v200_formulas():
    chosen_edges = {
        "energies": np.array([-3.0, -2.0], dtype=np.float32),
        "energy_traces": [
            np.array([-4.0, -9.0], dtype=np.float32),
            np.array([-1.0, -4.0], dtype=np.float32),
        ],
        "traces": [np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)],
    }
    normalized = normalize_edges_matlab_style(chosen_edges)
    assert np.allclose(
        normalized["edge_endpoint_energies"], np.array([-6.0, -2.0], dtype=np.float32)
    )
    assert np.allclose(normalized["energies"], np.array([-0.5, -1.0], dtype=np.float32))


@pytest.mark.unit
def test_matlab_crop_edges_v200_excludes_edges_that_expand_past_image_bounds():
    excluded = _matlab_crop_edges_v200(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        ],
        [np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)],
        [np.array([-2.0, -2.0], dtype=np.float32), np.array([-2.0, -2.0], dtype=np.float32)],
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        size_of_image=(3, 3, 3),
    )
    assert excluded.tolist() == [True, False]


@pytest.mark.unit
def test_prefilter_crop_aligns_matlab_voxel_spacing_to_zyx_traces():
    kept, cropped = prefilter_edge_indices_for_cleanup_matlab_style(
        [0],
        [
            np.array(
                [[2.0, 211.0, 142.0], [2.0, 211.0, 144.0], [2.0, 212.0, 148.0]],
                dtype=np.float32,
            )
        ],
        [np.array([16.0, 17.0, 15.0], dtype=np.float32)],
        [np.array([-273.0, -260.0, -300.0], dtype=np.float32)],
        lumen_radius_microns=np.array(
            [1.4] * 15 + [2.571732, 2.672696, 2.777624],
            dtype=np.float32,
        ),
        microns_per_voxel=np.array([0.916, 0.916, 1.99688], dtype=np.float32),
        size_of_image=(64, 256, 256),
    )

    assert kept == [0]
    assert cropped == 0


# ==============================================================================
# Parity and Routing (Consolidated from test_edge_parity_routing.py)
# ==============================================================================


@pytest.mark.unit
def test_use_watershed_discovery_requirements():
    # Exact-compatible origin + MATLAB grid alignment selects Watershed Discovery.
    assert _use_watershed_discovery(
        {"energy_origin": "python_native_hessian"},
        {"comparison_exact_network": True},
    )
    assert not _use_watershed_discovery(
        {"energy_origin": "python_pipeline"},
        {"comparison_exact_network": True},
    )
    assert _use_matlab_frontier_tracer is _use_watershed_discovery


@pytest.mark.unit
def test_discovery_strategy_aliases_match_domain_names():
    assert TracingDiscovery is MaintainedTracingDiscovery
    assert WatershedDiscovery is FrontierTracingDiscovery


@pytest.mark.unit
def test_append_candidate_unit_assigns_frontier_manifest_candidate_indices():
    target = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": {},
    }
    payload = {
        "candidate_source": "frontier",
        "traces": [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
        "connections": [[0, 1]],
        "metrics": [-1.0],
        "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
        "scale_traces": [np.zeros((2,), dtype=np.int16)],
        "origin_indices": [0],
        "connection_sources": ["frontier"],
        "frontier_lifecycle_events": [
            {
                "seed_origin_index": 0,
                "survived_candidate_manifest": True,
                "manifest_candidate_index": None,
            }
        ],
        "diagnostics": {},
    }
    _append_candidate_unit(target, payload)
    events = cast("list[dict[str, object]]", target["frontier_lifecycle_events"])
    assert events[0]["manifest_candidate_index"] == 0


@pytest.mark.unit
def test_candidate_manifest_reordered_and_endpoint_pairs():
    manifest = CandidateManifest(
        traces=[np.zeros((2, 3)), np.ones((2, 3))],
        connections=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
        metrics=np.asarray([2.0, 1.0], dtype=np.float64),
        energy_traces=[np.zeros(2), np.ones(2)],
        scale_traces=[np.zeros(2, dtype=np.int16), np.ones(2, dtype=np.int16)],
        origin_indices=np.asarray([0, 1], dtype=np.int32),
        connection_sources=["frontier", "watershed"],
    )

    reordered = manifest.reordered(np.asarray([1, 0], dtype=np.int32))
    assert reordered.metrics.tolist() == [1.0, 2.0]
    assert reordered.connection_sources == ["watershed", "frontier"]
    assert manifest.endpoint_pair_set() == {(0, 1), (2, 3)}


# ==============================================================================
# Graph and Cleanup
# ==============================================================================


@pytest.mark.unit
def test_cycle_cleanup_removes_worst_edge_per_cycle_component():
    connections = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], dtype=np.int32)
    keep = break_graph_cycles(connections)
    assert keep.tolist() == [True, True, False, True, True, False]


@pytest.mark.unit
def test_remove_short_hairs_repeats_until_graph_is_stable():
    adjacency_list = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    graph_edges = {
        (0, 1): np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        (1, 2): np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        (2, 3): np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float32),
    }
    _remove_short_hairs(graph_edges, adjacency_list, np.ones(3, dtype=np.float32), 1.5)
    assert not graph_edges


# ==============================================================================
# Matlab Style Helper Tests
# ==============================================================================


@pytest.mark.unit
def test_offset_coords_matlab_snaps_out_of_bounds():
    offsets = _construct_structuring_element_offsets_matlab(
        np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    coords = _offset_coords_matlab(np.array([0.0, 0.0, 0.0], dtype=np.float32), offsets, (3, 3, 3))
    assert np.all(coords >= 0)
    assert np.all(coords < 3)


@pytest.mark.unit
def test_matlab_edge_endpoint_positions_and_scales_use_trace_endpoints():
    trace = np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 5.0], [5.0, 7.0, 5.0]], dtype=np.float32)
    scale_trace = np.array([1, 2, 3], dtype=np.int16)
    start, _ = _matlab_edge_endpoint_positions_and_scales(trace, scale_trace)
    assert np.allclose(start[0], trace[0])
    assert start[1] == 1


# ==============================================================================
# Edge Selection and Conflicts (Consolidated from test_edge_cases.py)
# ==============================================================================


@pytest.mark.unit
def test_vertex_extraction_uses_matlab_paint_selection():
    energy = np.zeros((12, 12, 12), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    energy[5, 5, 5] = -5.0
    energy[6, 5, 5] = -4.0
    scale_indices[5, 5, 5] = 2
    scale_indices[6, 5, 5] = 2

    energy_data = EnergyResult.from_dict(
        {
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
    )
    params = {
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "length_dilation_ratio": 1.0,
    }
    vertices = extract_vertices(energy_data, params)
    assert len(vertices.positions) == 1
    assert np.allclose(vertices.positions[0], [5, 5, 5])


@pytest.mark.unit
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


@pytest.mark.unit
def test_choose_edges_tracks_conflict_provenance_by_source():
    vertex_positions = np.array([[1, 1, 1], [1, 5, 1], [1, 3, 1], [3, 5, 1]], dtype=np.float32)
    vertex_scales = np.zeros(4, dtype=np.int16)
    chosen_frontier = np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32)
    rejected_watershed = np.array(
        [[1, 3, 1], [1, 4, 1], [1, 5, 1], [2, 5, 1], [3, 5, 1]], dtype=np.float32
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
        {"number_of_edges_per_vertex": 4, "comparison_exact_network_use_conflict_painting": True},
    )
    assert chosen["connections"].tolist() == [[0, 1]]
    assert chosen["connection_sources"] == ["frontier"]


@pytest.mark.unit
def test_choose_edges_prefers_shorter_trace_when_metrics_are_equal():
    vertex_positions = np.array([[1, 1, 1], [1, 6, 1]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    shorter_trace = np.array([[1, 1, 1], [1, 4, 1], [1, 6, 1]], dtype=np.float32)
    candidates = {
        "traces": [np.zeros((4, 3)), shorter_trace],
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
    assert np.array_equal(chosen["traces"][0], shorter_trace)


@pytest.mark.unit
def test_choose_edges_exact_route_uses_seeded_trace_permutation(monkeypatch):
    seed_calls = []

    class FakeRng:
        def permutation(self, length):
            return np.arange(length - 1, -1, -1)

    monkeypatch.setattr(
        conflict_painting_module.np.random,
        "default_rng",
        lambda seed: seed_calls.append(seed) or FakeRng(),
    )

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
    _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4, "comparison_exact_network": True},
    )
    assert seed_calls == [conflict_painting_module.EXACT_ROUTE_CHOOSER_SEED]


@pytest.mark.unit
def test_snapshot_endpoint_influences_matlab_restores_overlap():
    painted_image = np.zeros((5, 5, 5), dtype=np.int32)
    painted_source_image = np.zeros((5, 5, 5), dtype=np.uint8)
    painted_image[1, 2, 1] = 7
    painted_source_image[1, 2, 1] = 2

    combined_coords, snapshot, _ = _snapshot_endpoint_influences_matlab(
        [np.array([[1, 2, 1]], dtype=np.int32)],
        painted_image,
        painted_source_image,
    )
    assert painted_image[1, 2, 1] == 0
    painted_image[combined_coords[:, 0], combined_coords[:, 1], combined_coords[:, 2]] = snapshot
    assert painted_image[1, 2, 1] == 7


@pytest.mark.unit
def test_choose_edges_uses_trace_endpoint_scales_for_vertex_influence():
    vertex_positions = np.array([[5, 5, 5], [5, 7, 5], [5, 9, 5], [5, 11, 5]], dtype=np.float32)
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
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]], dtype=np.float32
        ),
        (20, 20, 20),
        {
            "number_of_edges_per_vertex": 4,
            "sigma_per_influence_vertices": 1.0,
            "sigma_per_influence_edges": 0.5,
        },
    )
    assert chosen["connections"].tolist() == [[0, 1], [2, 3]]


def test_prune_orphan_edges_vectorized_parity():
    from slavv_python.pipeline.edges.cleanup import prune_orphan_edges

    # 3 vertices
    vertex_positions = np.array([[5, 5, 5], [10, 10, 10], [15, 15, 15]], dtype=np.float32)

    # 4 traces:
    # 1. Traces connecting vertex 0 to vertex 1 (Not orphan)
    # 2. Traces connecting vertex 1 to vertex 2 (Not orphan)
    # 3. Floating trace not touching any vertex or edge (Orphan)
    # 4. Trace connected to 3's interior but not to any vertex (Orphan - should be recursively pruned)
    traces = [
        np.array(
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]], dtype=np.float32
        ),
        np.array(
            [[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15]],
            dtype=np.float32,
        ),
        np.array([[20, 20, 20], [21, 21, 21], [22, 22, 22]], dtype=np.float32),
        np.array([[21, 21, 21], [30, 30, 30]], dtype=np.float32),
    ]

    volume_shape = (50, 50, 50)
    keep_mask = prune_orphan_edges(traces, volume_shape, vertex_positions)

    assert keep_mask.tolist() == [True, True, False, False]


# Made with Bob
