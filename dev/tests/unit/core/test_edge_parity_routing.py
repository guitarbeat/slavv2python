from __future__ import annotations

from typing import cast

import numpy as np
from source.core._edge_candidates.candidate_manifest import _append_candidate_unit
from source.core._edge_candidates.common import _use_matlab_frontier_tracer
from source.core._edges import standard as standard_edges


def test_use_matlab_frontier_tracer_requires_exact_network_and_matlab_energy_origin():
    assert not _use_matlab_frontier_tracer(
        {"energy_origin": "matlab_batch_hdf5"},
        {"comparison_exact_network": False},
    )
    assert not _use_matlab_frontier_tracer(
        {"energy_origin": "python_pipeline"},
        {"comparison_exact_network": True},
    )
    assert _use_matlab_frontier_tracer(
        {"energy_origin": "matlab_batch_hdf5"},
        {"comparison_exact_network": True},
    )


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
            },
            {
                "seed_origin_index": 0,
                "survived_candidate_manifest": False,
                "manifest_candidate_index": 99,
            },
        ],
        "diagnostics": {},
    }

    _append_candidate_unit(target, payload)

    events = cast("list[dict[str, object]]", target["frontier_lifecycle_events"])
    assert events[0]["manifest_candidate_index"] == 0
    assert events[1]["manifest_candidate_index"] is None


def test_standard_extract_edges_uses_matlab_frontier_branch_when_enabled():
    energy_data = {
        "energy": np.zeros((3, 3, 3), dtype=np.float32),
        "scale_indices": np.zeros((3, 3, 3), dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
        "energy_sign": -1.0,
        "energy_origin": "matlab_batch_hdf5",
    }
    vertices = {
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "scales": np.array([0, 0], dtype=np.int16),
    }
    params = {
        "comparison_exact_network": True,
        "microns_per_voxel": [1.0, 1.0, 1.0],
    }
    chosen = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "diagnostics": {"candidate_traced_edge_count": 0},
    }
    calls: list[str] = []

    def fake_generate_frontier(*args):
        calls.append("generate_frontier")
        del args
        return {"candidate_source": "frontier", "diagnostics": {}, "connections": np.zeros((0, 2))}

    def fake_finalize(*args):
        calls.append("finalize")
        return args[0]

    def fake_generate_fallback(*args):
        calls.append("generate_fallback")
        del args
        return {"candidate_source": "fallback", "diagnostics": {}, "connections": np.zeros((0, 2))}

    result = standard_edges.extract_edges(
        energy_data,
        vertices,
        params,
        empty_edges_result=lambda _vertex_positions: chosen,
        paint_vertex_center_image=lambda _positions, shape: np.zeros(shape, dtype=np.int32),
        use_matlab_frontier_tracer=lambda _energy_data, _params: True,
        generate_edge_candidates_matlab_frontier=fake_generate_frontier,
        finalize_matlab_parity_candidates=fake_finalize,
        generate_edge_candidates=fake_generate_fallback,
        choose_edges_for_workflow=lambda *_args: chosen,
        add_vertices_to_edges_matlab_style=lambda chosen_edges, *_args, **_kwargs: chosen_edges,
        finalize_edges_matlab_style=lambda chosen_edges, **_kwargs: chosen_edges,
    )

    assert result is chosen
    assert calls == ["generate_frontier", "finalize"]


