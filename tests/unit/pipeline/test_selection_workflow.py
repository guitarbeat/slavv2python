"""Tests for the post-Edge Discovery selection workflow module."""

from __future__ import annotations

import numpy as np

from slavv_python.pipeline.edges.selection_workflow import select_and_finalize_edge_set
from slavv_python.schema.results import EnergyResult, VertexSet


def test_select_and_finalize_edge_set_runs_choose_bridge_finalize(monkeypatch) -> None:
    energy = EnergyResult.from_dict(
        {
            "energy": np.zeros((4, 4, 4), dtype=np.float64),
            "scale_indices": np.zeros((4, 4, 4), dtype=np.int16),
            "lumen_radius_microns": np.array([1.0], dtype=np.float64),
            "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float64),
            "energy_sign": -1.0,
            "energy_origin": "python_native_hessian",
        }
    )
    vertices = VertexSet.from_dict(
        {
            "positions": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
            "scales": np.array([0, 0], dtype=np.int16),
        }
    )
    params = {
        "comparison_exact_network": True,
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "number_of_edges_per_vertex": 4,
    }
    candidates = {
        "traces": [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)],
        "connections": np.array([[0, 1]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float64),
        "energy_traces": [np.array([-1.0], dtype=np.float64)],
        "scale_traces": [np.array([0], dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "connection_sources": ["frontier"],
        "diagnostics": {},
    }
    stages: list[str] = []

    def fake_choose(*_a, **_k):
        stages.append("choose")
        return {
            "traces": candidates["traces"],
            "connections": candidates["connections"],
            "energies": np.array([-1.0], dtype=np.float64),
            "energy_traces": candidates["energy_traces"],
            "scale_traces": candidates["scale_traces"],
            "diagnostics": {"candidate_traced_edge_count": 1},
        }

    def fake_bridge(selected, *_a, **_k):
        stages.append("bridge")
        return selected

    def fake_finalize(selected, **_k):
        stages.append("finalize")
        return selected

    monkeypatch.setattr(
        "slavv_python.pipeline.edges.selection_workflow.choose_edges_for_workflow",
        fake_choose,
    )
    monkeypatch.setattr(
        "slavv_python.pipeline.edges.selection_workflow.add_vertices_to_edges_matlab_style",
        fake_bridge,
    )
    monkeypatch.setattr(
        "slavv_python.pipeline.edges.selection_workflow.finalize_edges_matlab_style",
        fake_finalize,
    )

    result = select_and_finalize_edge_set(candidates, energy, vertices, params)

    assert stages == ["choose", "bridge", "finalize"]
    assert len(result.traces) == 1
    assert np.array_equal(result.connections, [[0, 1]])
    np.testing.assert_allclose(result.extra["lumen_radius_microns"], [1.0])


def test_select_and_finalize_skips_bridge_when_disabled(monkeypatch) -> None:
    energy = EnergyResult.from_dict(
        {
            "energy": np.zeros((3, 3, 3), dtype=np.float64),
            "scale_indices": np.zeros((3, 3, 3), dtype=np.int16),
            "lumen_radius_microns": np.array([1.0], dtype=np.float64),
            "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float64),
            "energy_sign": -1.0,
        }
    )
    vertices = VertexSet.from_dict(
        {
            "positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            "scales": np.array([0], dtype=np.int16),
        }
    )
    params = {"comparison_exact_network": False, "microns_per_voxel": [1.0, 1.0, 1.0]}
    stages: list[str] = []

    monkeypatch.setattr(
        "slavv_python.pipeline.edges.selection_workflow.choose_edges_for_workflow",
        lambda *_a, **_k: {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "energies": np.zeros(0, dtype=np.float64),
            "energy_traces": [],
            "scale_traces": [],
            "diagnostics": {},
        },
    )
    monkeypatch.setattr(
        "slavv_python.pipeline.edges.selection_workflow.add_vertices_to_edges_matlab_style",
        lambda selected, *_a, **_k: stages.append("bridge") or selected,
    )
    monkeypatch.setattr(
        "slavv_python.pipeline.edges.selection_workflow.finalize_edges_matlab_style",
        lambda selected, **_k: stages.append("finalize") or selected,
    )

    select_and_finalize_edge_set(
        {"connections": np.zeros((0, 2)), "traces": []},
        energy,
        vertices,
        params,
        apply_bridge_vertices=False,
    )
    assert stages == ["finalize"]
