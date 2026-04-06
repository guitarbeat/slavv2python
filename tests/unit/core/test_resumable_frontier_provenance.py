import joblib
import numpy as np
import pytest

from slavv.core.tracing import extract_edges_resumable
from slavv.runtime.run_state import RunContext


@pytest.mark.unit
def test_extract_edges_resumable_preserves_frontier_candidate_provenance(monkeypatch, tmp_path):
    energy = -np.ones((5, 5, 5), dtype=np.float64)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "energy_sign": -1.0,
        "energy_origin": "matlab_batch_hdf5",
    }
    vertices = {
        "positions": np.array([[2.0, 1.0, 2.0], [2.0, 3.0, 2.0]], dtype=np.float32),
        "scales": np.array([0, 0], dtype=np.int16),
    }
    params = {
        "comparison_exact_network": True,
        "number_of_edges_per_vertex": 1,
        "parity_frontier_reachability_gate": False,
    }

    def fake_frontier(*args, **kwargs):
        origin_index = int(args[7])
        return {
            "origin_index": origin_index,
            "candidate_source": "frontier",
            "traces": [
                np.array(
                    [
                        vertices["positions"][origin_index],
                        vertices["positions"][1 - origin_index],
                    ],
                    dtype=np.float32,
                )
            ],
            "connections": [[origin_index, 1 - origin_index]],
            "metrics": [-5.0],
            "energy_traces": [np.array([-5.0, -5.0], dtype=np.float32)],
            "scale_traces": [np.array([0, 0], dtype=np.int16)],
            "origin_indices": [origin_index],
            "connection_sources": ["frontier"],
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

    def fake_supplement(candidates, *args, **kwargs):
        return candidates

    def fake_choose(candidates, *args, **kwargs):
        return {
            "traces": list(candidates["traces"]),
            "connections": np.asarray(candidates["connections"], dtype=np.int32),
            "energies": np.asarray(candidates["metrics"], dtype=np.float32),
            "diagnostics": dict(candidates["diagnostics"]),
        }

    monkeypatch.setattr("slavv.core.tracing._trace_origin_edges_matlab_frontier", fake_frontier)
    monkeypatch.setattr(
        "slavv.core.tracing._supplement_matlab_frontier_candidates_with_watershed_joins",
        fake_supplement,
    )
    monkeypatch.setattr("slavv.core.tracing._choose_edges_matlab_style", fake_choose)

    run_context = RunContext(run_dir=tmp_path / "run", target_stage="edges")
    edges = extract_edges_resumable(energy_data, vertices, params, run_context.stage("edges"))

    assert edges["connections"].tolist() == [[0, 1], [1, 0]]

    candidates = joblib.load(run_context.stage("edges").artifact_path("candidates.pkl"))
    assert candidates["connection_sources"] == ["frontier", "frontier"]

    candidate_audit = (
        run_context.stage("edges").artifact_path("candidate_audit.json").read_text(encoding="utf-8")
    )
    assert '"frontier_only_pair_count": 1' in candidate_audit
    assert '"fallback_only_pair_count": 0' in candidate_audit
