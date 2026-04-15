from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib

from slavv.parity.comparison import orchestrate_comparison

if TYPE_CHECKING:
    from pathlib import Path


def test_orchestrate_comparison_respects_python_result_source(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

    # Setup directories
    run_root = tmp_path / "run_root"
    run_root.mkdir()

    # 02_Output/python_results/checkpoints
    python_results_dir = run_root / "02_Output" / "python_results"
    checkpoint_dir = python_results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    # Create dummy checkpoints (vertices, edges, and network are required)
    joblib.dump({"positions": [[9, 9, 9]]}, checkpoint_dir / "checkpoint_vertices.pkl")
    joblib.dump({"traces": []}, checkpoint_dir / "checkpoint_edges.pkl")
    joblib.dump({"strands": []}, checkpoint_dir / "checkpoint_network.pkl")

    # Create a dummy network.json
    (python_results_dir / "network.json").write_text(
        json.dumps(
            {
                "vertices": {"positions": [[1, 2, 3]]},
                "edges": {"traces": []},
                "network": {"strands": []},
            }
        ),
        encoding="utf-8",
    )

    # Setup dummy MATLAB results to avoid failures
    matlab_results_dir = run_root / "02_Output" / "matlab_results"
    batch_dir = matlab_results_dir / "batch_test"
    batch_dir.mkdir(parents=True)
    (batch_dir / "network_test.vmv").touch()  # Minimal file for discovery

    # Mocking dependencies
    def fake_evaluate_preflight(*args, **kwargs):
        from slavv.parity.preflight import OutputRootPreflightReport

        return OutputRootPreflightReport(
            output_root=str(run_root), preflight_status="passed", allows_launch=True
        )

    def fake_inspect_matlab_status(*args, **kwargs):
        from slavv.parity.matlab_status import MatlabStatusReport

        return MatlabStatusReport(output_directory=str(matlab_results_dir))

    def fake_run_matlab(*args, **kwargs):
        return {"success": True, "batch_folder": str(batch_dir)}

    def fake_run_python(*args, **kwargs):
        return {"success": True}

    def fake_load_matlab_results(_batch_folder):
        return {
            "vertices": {"positions": []},
            "edges": {"traces": []},
            "network": {"strands": []},
            "network_stats": {"strand_count": 0},
        }

    captured = {}

    def fake_compare_results(_matlab, python, _depth):
        # Handle both run and load structures
        if "comparison_mode" in python:
            captured["python_source"] = python["comparison_mode"].get(
                "result_source", "native_python"
            )
        else:
            captured["python_source"] = "unknown"

        return {
            "matlab": {"vertices_count": 0, "edges_count": 0, "strand_count": 0},
            "python": {"vertices_count": 0, "edges_count": 0, "strand_count": 0},
            "performance": {"matlab_runtime": 0.0, "python_runtime": 0.0},
            "vertices": {"matlab_count": 0, "python_count": 0},
            "edges": {"matlab_count": 0, "python_count": 0},
            "network": {"matlab_strand_count": 0, "python_strand_count": 0},
            "parity_gate": {"vertices_exact": True, "edges_exact": True, "strands_exact": True},
        }

    monkeypatch.setattr(
        comparison_module, "evaluate_output_root_preflight", fake_evaluate_preflight
    )
    monkeypatch.setattr(comparison_module, "inspect_matlab_status", fake_inspect_matlab_status)
    monkeypatch.setattr(comparison_module, "run_matlab_vectorization", fake_run_matlab)
    monkeypatch.setattr(comparison_module, "run_python_vectorization", fake_run_python)
    monkeypatch.setattr(comparison_module, "import_matlab_batch", lambda *args, **kwargs: [])
    monkeypatch.setattr(comparison_module, "load_matlab_batch_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(comparison_module, "load_matlab_batch_results", fake_load_matlab_results)
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)
    monkeypatch.setattr(comparison_module, "generate_summary", lambda *args: None)
    monkeypatch.setattr(comparison_module, "generate_manifest", lambda *args: None)

    # Test with network-json-only and skip_python=True to trigger loading
    orchestrate_comparison(
        input_file="test.tif",
        output_dir=run_root,
        matlab_path="matlab",
        project_root=tmp_path,
        params={},
        skip_python=True,
        python_result_source="network-json-only",
    )

    assert captured["python_source"] == "network_json"

    # Test with checkpoints-only and skip_python=True
    orchestrate_comparison(
        input_file="test.tif",
        output_dir=run_root,
        matlab_path="matlab",
        project_root=tmp_path,
        params={},
        skip_python=True,
        python_result_source="checkpoints-only",
    )

    assert captured["python_source"] == "checkpoints"
