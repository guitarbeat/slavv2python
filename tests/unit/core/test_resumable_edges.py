from __future__ import annotations

import joblib
import numpy as np

from slavv_python.core.edges_internal import resumable_edges
from slavv_python.runtime import RunContext


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

    def fake_generate_edge_candidates(*args, **kwargs):
        calls["generate_args"] = args
        calls["generate_kwargs"] = kwargs
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
        empty_edges_result=lambda _vertex_positions: {
            "traces": [],
            "connections": np.zeros((0, 2)),
        },
        build_edge_candidate_audit=fake_build_edge_candidate_audit,
        build_frontier_candidate_lifecycle=lambda *_args: {"events": []},
        finalize_matlab_parity_candidates=lambda *_args: candidates,
        normalize_candidate_origin_counts=lambda raw: {
            int(key): int(value) for key, value in raw.items()
        },
        generate_edge_candidates_matlab_frontier=lambda *_args: candidates,
        generate_edge_candidates=fake_generate_edge_candidates,
        choose_edges_for_workflow=fake_choose_edges_for_workflow,
        add_vertices_to_edges_matlab_style=lambda chosen, *_args, **_kwargs: chosen,
        finalize_edges_matlab_style=lambda chosen, **_kwargs: chosen,
        paint_vertex_center_image=lambda _positions, shape: np.zeros(shape, dtype=np.int32),
        paint_vertex_image=lambda _positions, _scales, _radii, shape: np.zeros(
            shape, dtype=np.int32
        ),
        use_matlab_frontier_tracer=lambda *_args: False,
    )

    assert result is chosen
    assert "generate_args" in calls
    assert "generate_kwargs" in calls
    assert "choose_args" in calls
    assert calls["audit"] == {
        "candidate_payload": candidates,
        "vertex_count": 2,
        "use_frontier_tracer": False,
        "frontier_origin_counts": {0: 1},
        "supplement_origin_counts": {0: 2},
    }
    assert calls["generate_kwargs"]["vertex_image"] is not None
    assert stage_controller.artifact_path("candidates.pkl").is_file()
    assert stage_controller.artifact_path("chosen_edges.pkl").is_file()
    assert stage_controller.artifact_path("candidate_audit.json").is_file()


def test_extract_edges_resumable_uses_matlab_frontier_branch_when_enabled(tmp_path):
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="edges")
    stage_controller = run_context.stage("edges")

    energy_data = {
        "energy": np.zeros((3, 3, 3), dtype=np.float32),
        "scale_indices": np.zeros((3, 3, 3), dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
        "energy_sign": -1.0,
        "energy_origin": "python_native_hessian",
    }
    vertices = {
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "scales": np.array([0, 0], dtype=np.int16),
    }
    params = {
        "comparison_exact_network": True,
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "number_of_edges_per_vertex": 4,
    }

    frontier_candidates = {
        "traces": [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
        "connections": np.array([[0, 1]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float32),
        "energy_traces": [np.array([-1.0], dtype=np.float32)],
        "scale_traces": [np.array([0], dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "connection_sources": ["frontier"],
        "frontier_lifecycle_events": [
            {"seed_origin_index": 0, "survived_candidate_manifest": True}
        ],
        "diagnostics": {
            "frontier_per_origin_candidate_counts": {"0": 1},
            "watershed_per_origin_candidate_counts": {"0": 1},
        },
    }
    chosen = {
        "traces": frontier_candidates["traces"],
        "connections": frontier_candidates["connections"],
        "energies": np.array([-1.0], dtype=np.float32),
        "energy_traces": frontier_candidates["energy_traces"],
        "scale_traces": frontier_candidates["scale_traces"],
        "vertex_positions": vertices["positions"],
        "connection_sources": ["frontier"],
        "chosen_candidate_indices": np.array([0], dtype=np.int32),
        "diagnostics": {"candidate_traced_edge_count": 1},
    }
    calls: list[str] = []

    def fake_finalize(*_args):
        calls.append("finalize")
        return frontier_candidates

    def fake_generate_frontier(*_args, **kwargs):
        calls.append("generate_frontier")
        heartbeat = kwargs.get("heartbeat")
        if callable(heartbeat):
            heartbeat(7, len(frontier_candidates["connections"]))
        return frontier_candidates

    def fake_generate_fallback(*_args, **_kwargs):
        calls.append("generate_fallback")
        return frontier_candidates

    result = resumable_edges.extract_edges_resumable(
        energy_data,
        vertices,
        params,
        stage_controller,
        atomic_joblib_dump=lambda value, path: joblib.dump(value, path),
        empty_edges_result=lambda _vertex_positions: {
            "traces": [],
            "connections": np.zeros((0, 2)),
        },
        build_edge_candidate_audit=lambda *_args, **_kwargs: {"audit": True},
        build_frontier_candidate_lifecycle=lambda *_args: {"events": [1]},
        finalize_matlab_parity_candidates=fake_finalize,
        normalize_candidate_origin_counts=lambda raw: {
            int(key): int(value) for key, value in raw.items()
        },
        generate_edge_candidates_matlab_frontier=fake_generate_frontier,
        generate_edge_candidates=fake_generate_fallback,
        choose_edges_for_workflow=lambda *_args: chosen,
        add_vertices_to_edges_matlab_style=lambda chosen, *_args, **_kwargs: chosen,
        finalize_edges_matlab_style=lambda chosen, **_kwargs: chosen,
        paint_vertex_center_image=lambda _positions, shape: np.zeros(shape, dtype=np.int32),
        paint_vertex_image=lambda _positions, _scales, _radii, shape: np.zeros(
            shape, dtype=np.int32
        ),
        use_matlab_frontier_tracer=lambda *_args: True,
    )

    assert result is chosen
    assert calls == ["generate_frontier", "finalize"]
    assert stage_controller.artifact_path("candidate_lifecycle.json").is_file()
    candidate_checkpoint_path = run_context.checkpoints_dir / "checkpoint_edge_candidates.pkl"
    assert candidate_checkpoint_path.is_file()
    candidate_checkpoint = joblib.load(candidate_checkpoint_path)
    assert candidate_checkpoint["connections"].tolist() == [[0, 1]]
    assert candidate_checkpoint["diagnostics"]["frontier_per_origin_candidate_counts"] == {"0": 1}
