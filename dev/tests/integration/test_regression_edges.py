from unittest.mock import patch

import numpy as np

import slavv.core.edges as edges_module
from slavv.core import SLAVVProcessor
from slavv.core.edge_candidates import (
    _matlab_frontier_edge_budget,
    _matlab_parity_frontier_budget_mode,
    _parity_watershed_candidate_mode,
)


@patch(
    "slavv.core.edge_candidates.estimate_vessel_directions",
    return_value=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float),
)
def test_extract_edges_regression(mock_generate_directions):
    size = 21
    coords = np.indices((size, size, size))
    x = coords[1] - size // 2
    z = coords[2] - size // 2
    energy = -(x**2 + z**2).astype(float)

    energy_data = {
        "energy": energy,
        "lumen_radius_pixels": np.array([2.0], dtype=float),
        "lumen_radius_microns": np.array([2.0], dtype=float),
        "lumen_radius_pixels_axes": np.array([[2.0, 2.0, 2.0]], dtype=float),
        "energy_sign": -1.0,
    }
    vertices = {
        "positions": np.array([[10.0, 10.0, 10.0]], dtype=float),
        "scales": np.array([0], dtype=int),
    }
    params = {
        "number_of_edges_per_vertex": 2,
        "step_size_per_origin_radius": 2.0,
        "length_dilation_ratio": 5.0,
        "microns_per_voxel": [1.0, 1.0, 1.0],
    }

    processor = SLAVVProcessor()
    edges = processor.extract_edges(energy_data, vertices, params)
    diagnostics = edges["diagnostics"]

    assert edges["connections"].shape == (0, 2)
    assert len(edges["traces"]) == 0
    assert diagnostics["candidate_traced_edge_count"] == 2
    assert diagnostics["dangling_edge_count"] == 2
    assert diagnostics["chosen_edge_count"] == 0


@patch(
    "slavv.core.edge_candidates.estimate_vessel_directions",
    return_value=np.array([[0.0, 1.0, 0.0]], dtype=float),
)
def test_extract_edges_recovers_near_terminal_attachment(mock_generate_directions):
    energy = -np.ones((7, 7, 7), dtype=float)

    energy_data = {
        "energy": energy,
        "lumen_radius_pixels": np.array([1.0], dtype=float),
        "lumen_radius_microns": np.array([1.1], dtype=float),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=float),
        "energy_sign": -1.0,
    }
    vertices = {
        "positions": np.array([[3.0, 0.0, 3.0], [3.0, 4.0, 3.0]], dtype=float),
        "scales": np.array([0, 0], dtype=int),
    }
    params = {
        "number_of_edges_per_vertex": 1,
        "step_size_per_origin_radius": 1.5,
        "max_edge_length_per_origin_radius": 3.0,
        "microns_per_voxel": [1.0, 1.0, 1.0],
    }

    processor = SLAVVProcessor()
    edges = processor.extract_edges(energy_data, vertices, params)
    diagnostics = edges["diagnostics"]

    assert edges["connections"].tolist() == [[0, 1]]
    assert np.allclose(edges["traces"][0][-1], vertices["positions"][1])
    assert diagnostics["candidate_traced_edge_count"] == 2
    assert diagnostics["terminal_edge_count"] == 1
    assert diagnostics["dangling_edge_count"] == 1
    assert diagnostics["chosen_edge_count"] == 1
    assert diagnostics["terminal_reverse_near_hit_count"] == 1
    assert diagnostics["stop_reason_counts"]["max_steps"] == 1
    assert diagnostics["stop_reason_counts"]["bounds"] == 1


def test_extract_edges_routes_matlab_energy_parity_runs_to_frontier(monkeypatch):
    energy = -np.ones((5, 5, 5), dtype=float)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=float),
        "lumen_radius_microns": np.array([1.0], dtype=float),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=float),
        "energy_sign": -1.0,
        "energy_origin": "matlab_batch_hdf5",
    }
    vertices = {
        "positions": np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=float),
        "scales": np.array([0, 0], dtype=int),
    }

    captured = {"called": False}

    def fake_frontier(*args, **kwargs):
        captured["called"] = True
        return {
            "traces": [np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=np.float32)],
            "connections": np.array([[0, 1]], dtype=np.int32),
            "metrics": np.array([-5.0], dtype=np.float32),
            "energy_traces": [np.array([-5.0, -5.0], dtype=np.float32)],
            "scale_traces": [np.array([0, 0], dtype=np.int16)],
            "origin_indices": np.array([0], dtype=np.int32),
            "diagnostics": {
                "candidate_traced_edge_count": 0,
                "terminal_edge_count": 0,
                "self_edge_count": 0,
                "duplicate_directed_pair_count": 0,
                "antiparallel_pair_count": 0,
                "chosen_edge_count": 0,
                "dangling_edge_count": 0,
                "negative_energy_rejected_count": 0,
                "conflict_rejected_count": 0,
                "degree_pruned_count": 0,
                "orphan_pruned_count": 0,
                "cycle_pruned_count": 0,
                "terminal_direct_hit_count": 1,
                "terminal_reverse_center_hit_count": 0,
                "terminal_reverse_near_hit_count": 0,
                "stop_reason_counts": {
                    "bounds": 0,
                    "nan": 0,
                    "energy_threshold": 0,
                    "energy_rise_step_halving": 0,
                    "max_steps": 0,
                    "direct_terminal_hit": 0,
                    "frontier_exhausted_nonnegative": 0,
                    "length_limit": 0,
                    "terminal_frontier_hit": 1,
                },
            },
        }

    monkeypatch.setattr(
        edges_module,
        "_generate_edge_candidates_matlab_frontier",
        fake_frontier,
    )

    processor = SLAVVProcessor()
    edges = processor.extract_edges(
        energy_data,
        vertices,
        {"comparison_exact_network": True, "number_of_edges_per_vertex": 1},
    )

    assert captured["called"]
    assert edges["connections"].tolist() == [[0, 1]]


def test_extract_edges_forces_matlab_v300_budget_through_imported_matlab_workflow(
    monkeypatch,
):
    energy = -np.ones((5, 5, 5), dtype=float)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=float),
        "lumen_radius_microns": np.array([1.0], dtype=float),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=float),
        "energy_sign": -1.0,
        "energy_origin": "matlab_batch_hdf5",
    }
    vertices = {
        "positions": np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=float),
        "scales": np.array([0, 0], dtype=int),
    }
    params = {"comparison_exact_network": True, "number_of_edges_per_vertex": 4}
    captured: dict[str, dict[str, object]] = {}

    def fake_frontier(
        energy,
        scale_indices,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        vertex_center_image,
        workflow_params,
    ):
        del energy
        del scale_indices
        del vertex_positions
        del vertex_scales
        del lumen_radius_microns
        del microns_per_voxel
        del vertex_center_image
        captured["frontier"] = workflow_params
        return {
            "traces": [np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=np.float32)],
            "connections": np.array([[0, 1]], dtype=np.int32),
            "metrics": np.array([-5.0], dtype=np.float32),
            "energy_traces": [np.array([-5.0, -5.0], dtype=np.float32)],
            "scale_traces": [np.array([0, 0], dtype=np.int16)],
            "origin_indices": np.array([0], dtype=np.int32),
            "connection_sources": ["frontier"],
            "diagnostics": {},
        }

    def fake_finalize(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        workflow_params,
        microns_per_voxel,
    ):
        del candidates
        del energy
        del scale_indices
        del vertex_positions
        del energy_sign
        del microns_per_voxel
        captured["finalize"] = workflow_params
        return {
            "traces": [np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=np.float32)],
            "connections": np.array([[0, 1]], dtype=np.int32),
            "metrics": np.array([-5.0], dtype=np.float32),
            "energy_traces": [np.array([-5.0, -5.0], dtype=np.float32)],
            "scale_traces": [np.array([0, 0], dtype=np.int16)],
            "origin_indices": np.array([0], dtype=np.int32),
            "connection_sources": ["frontier"],
            "diagnostics": {},
        }

    def fake_choose(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_pixels_axes,
        image_shape,
        workflow_params,
    ):
        del candidates
        del vertex_positions
        del vertex_scales
        del lumen_radius_pixels_axes
        del image_shape
        captured["choose"] = workflow_params
        return {
            "traces": [np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=np.float32)],
            "connections": np.array([[0, 1]], dtype=np.int32),
            "metrics": np.array([-5.0], dtype=np.float32),
            "energy_traces": [np.array([-5.0, -5.0], dtype=np.float32)],
            "scale_traces": [np.array([0, 0], dtype=np.int16)],
            "diagnostics": {"candidate_traced_edge_count": 1},
        }

    monkeypatch.setattr(edges_module, "_generate_edge_candidates_matlab_frontier", fake_frontier)
    monkeypatch.setattr(edges_module, "_finalize_matlab_parity_candidates", fake_finalize)
    monkeypatch.setattr(edges_module, "choose_edges_for_workflow", fake_choose)

    processor = SLAVVProcessor()
    edges = processor.extract_edges(energy_data, vertices, params)

    assert edges["connections"].tolist() == [[0, 1]]
    assert _matlab_frontier_edge_budget(params) == 4
    assert _parity_watershed_candidate_mode(params) == "all_contacts"
    for key in ("frontier", "finalize", "choose"):
        workflow_params = captured[key]
        assert workflow_params is not params
        assert workflow_params["number_of_edges_per_vertex"] == 4
        assert _matlab_frontier_edge_budget(workflow_params) == 2
        assert _matlab_parity_frontier_budget_mode(workflow_params) == (
            "accepted_candidates"
        )
        assert _parity_watershed_candidate_mode(workflow_params) == (
            "remaining_origin_contacts"
        )
