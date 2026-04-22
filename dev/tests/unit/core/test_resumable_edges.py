from __future__ import annotations

import joblib
import numpy as np

from slavv.core._edges import resumable as resumable_edges
from slavv.runtime import RunContext


def test_extract_edges_resumable_uses_maintained_candidate_generator(tmp_path):
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="edges")
    stage_controller = run_context.stage("edges")

    energy_data = {
        "energy": np.zeros((3, 3, 3), dtype=np.float32),
        "scale_indices": np.zeros((3, 3, 3), dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
        "energy_sign": -1.0,
    }
    vertices = {
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "scales": np.array([0, 0], dtype=np.int16),
    }
    params = {
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "number_of_edges_per_vertex": 4,
    }

    calls: dict[str, object] = {}
    candidates = {
        "traces": [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
        "connections": np.array([[0, 1]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float32),
        "energy_traces": [np.array([-1.0], dtype=np.float32)],
        "scale_traces": [np.array([0], dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "connection_sources": ["frontier"],
        "diagnostics": {
            "watershed_per_origin_candidate_counts": {"0": 2},
        },
    }
    chosen = {
        "traces": candidates["traces"],
        "connections": candidates["connections"],
        "energies": np.array([-1.0], dtype=np.float32),
        "energy_traces": candidates["energy_traces"],
        "scale_traces": candidates["scale_traces"],
        "vertex_positions": vertices["positions"],
        "connection_sources": ["frontier"],
        "diagnostics": {"candidate_traced_edge_count": 1},
    }

    def fake_generate_edge_candidates(*args):
        calls["generate_args"] = args
        return candidates

    def fake_build_edge_candidate_audit(
        candidate_payload,
        vertex_count,
        use_frontier_tracer,
        frontier_origin_counts,
        supplement_origin_counts,
    ):
        calls["audit"] = {
            "candidate_payload": candidate_payload,
            "vertex_count": vertex_count,
            "use_frontier_tracer": use_frontier_tracer,
            "frontier_origin_counts": frontier_origin_counts,
            "supplement_origin_counts": supplement_origin_counts,
        }
        return {"audit": True}

    def fake_choose_edges_for_workflow(*args):
        calls["choose_args"] = args
        return chosen

    result = resumable_edges.extract_edges_resumable(
        energy_data,
        vertices,
        params,
        stage_controller,
        atomic_joblib_dump=lambda value, path: joblib.dump(value, path),
        empty_edges_result=lambda _vertex_positions: {"traces": [], "connections": np.zeros((0, 2))},
        build_edge_candidate_audit=fake_build_edge_candidate_audit,
        normalize_candidate_origin_counts=lambda raw: {int(key): int(value) for key, value in raw.items()},
        generate_edge_candidates=fake_generate_edge_candidates,
        choose_edges_for_workflow=fake_choose_edges_for_workflow,
        paint_vertex_center_image=lambda _positions, shape: np.zeros(shape, dtype=np.int32),
    )

    assert result is chosen
    assert "generate_args" in calls
    assert "choose_args" in calls
    assert calls["audit"] == {
        "candidate_payload": candidates,
        "vertex_count": 2,
        "use_frontier_tracer": True,
        "frontier_origin_counts": {0: 1},
        "supplement_origin_counts": {0: 2},
    }
    assert stage_controller.artifact_path("candidates.pkl").is_file()
    assert stage_controller.artifact_path("chosen_edges.pkl").is_file()
    assert stage_controller.artifact_path("candidate_audit.json").is_file()
