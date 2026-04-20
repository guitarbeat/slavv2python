import json
from pathlib import Path

from slavv.parity.comparison import orchestrate_comparison, run_standalone_comparison
from slavv.parity.matlab_status import MatlabStatusReport
from slavv.parity.preflight import OutputRootPreflightReport


def _write_python_archive_surface(python_dir: Path) -> dict[str, object]:
    checkpoint_dir = python_dir / "checkpoints"
    vertices_stage = python_dir / "stages" / "vertices"
    edges_stage = python_dir / "stages" / "edges"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vertices_stage.mkdir(parents=True, exist_ok=True)
    edges_stage.mkdir(parents=True, exist_ok=True)

    (python_dir / "network.json").write_text(
        json.dumps(
            {
                "vertices": {"positions": [[0, 0, 0]]},
                "edges": {"connections": [[0, 0]], "traces": []},
                "network": {"strands": []},
            }
        ),
        encoding="utf-8",
    )
    (python_dir / "python_comparison_parameters.json").write_text("{}", encoding="utf-8")
    (python_dir / "network.vmv").write_text("vmv", encoding="utf-8")
    (python_dir / "network.casx").write_text("casx", encoding="utf-8")
    (python_dir / "network_vertices.csv").write_text("vertices", encoding="utf-8")
    (python_dir / "network_edges.csv").write_text("edges", encoding="utf-8")
    (checkpoint_dir / "checkpoint_energy.pkl").write_bytes(b"energy")
    (checkpoint_dir / "checkpoint_vertices.pkl").write_bytes(b"vertices")
    (checkpoint_dir / "checkpoint_edges.pkl").write_bytes(b"edges")
    (checkpoint_dir / "checkpoint_network.pkl").write_bytes(b"network")
    (vertices_stage / "stage_manifest.json").write_text("{}", encoding="utf-8")
    (edges_stage / "stage_manifest.json").write_text("{}", encoding="utf-8")
    (edges_stage / "candidate_audit.json").write_text("{}", encoding="utf-8")
    (edges_stage / "candidate_lifecycle.json").write_text("{}", encoding="utf-8")
    (edges_stage / "candidates.pkl").write_bytes(b"candidates")

    return {
        "success": True,
        "elapsed_time": 0.0,
        "output_dir": str(python_dir),
        "vertices_count": 1,
        "edges_count": 0,
        "network_strands_count": 0,
        "results": {
            "vertices": {"positions": [[0, 0, 0]]},
            "edges": {"connections": [[0, 0]], "traces": []},
            "network": {"strands": []},
        },
        "comparison_mode": {"result_source": "network_json", "energy_source": "native_python"},
    }


def _write_matlab_archive_surface(matlab_dir: Path) -> Path:
    batch_dir = matlab_dir / "batch_260418-120000"
    data_dir = batch_dir / "data"
    vectors_dir = batch_dir / "vectors"
    settings_dir = batch_dir / "settings"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)

    (matlab_dir / "matlab_run.log").write_text("ok", encoding="utf-8")
    (data_dir / "energy_testROI").write_bytes(b"x" * 1_100_000)
    (vectors_dir / "vertices_testROI.mat").write_bytes(b"vertices")
    (vectors_dir / "curated_vertices_testROI.mat").write_bytes(b"curated_vertices")
    (vectors_dir / "edges_testROI.mat").write_bytes(b"edges")
    (vectors_dir / "curated_edges_testROI.mat").write_bytes(b"curated_edges")
    (vectors_dir / "network_testROI.mat").write_bytes(b"network")
    (vectors_dir / "preview_projection.tif").write_bytes(b"preview")
    (settings_dir / "batch.mat").write_bytes(b"settings")
    (settings_dir / "energy_testROI.mat").write_bytes(b"energy_settings")
    return batch_dir


def _make_preflight_report(output_root: Path) -> OutputRootPreflightReport:
    return OutputRootPreflightReport(
        output_root=str(output_root),
        resolved_output_root=str(output_root.resolve()),
        preflight_status="passed",
        allows_launch=True,
        writable=True,
        output_root_exists=True,
        output_root_created=False,
        free_space_gb=20.0,
        required_space_gb=5.0,
        recommended_action="Proceed.",
    )


def _make_status_report(matlab_dir: Path, *, batch_dir: Path | None) -> MatlabStatusReport:
    return MatlabStatusReport(
        output_directory=str(matlab_dir),
        matlab_resume_state_file=str(matlab_dir / "matlab_resume_state.json"),
        matlab_log_file=str(matlab_dir / "matlab_run.log"),
        matlab_batch_folder=str(batch_dir) if batch_dir is not None else "",
        matlab_batch_complete=batch_dir is not None,
        matlab_resume_mode="complete-noop" if batch_dir is not None else "fresh",
        matlab_last_completed_stage="network" if batch_dir is not None else "",
        matlab_next_stage="" if batch_dir is not None else "energy",
        matlab_rerun_prediction=(
            "Managed archive is analysis-only."
            if batch_dir is not None
            else "No reusable MATLAB batch found; rerun will start at energy."
        ),
    )


def _comparison_report() -> dict[str, object]:
    return {
        "matlab": {"elapsed_time": 2.0, "vertices_count": 1},
        "python": {"elapsed_time": 1.0, "vertices_count": 1},
        "performance": {"speedup": 2.0, "faster": "Python"},
        "vertices": {"matlab_count": 1, "python_count": 1, "matches_exactly": True},
        "edges": {
            "matlab_count": 0,
            "python_count": 0,
            "matches_exactly": True,
            "diagnostics": {
                "candidate_audit": {
                    "schema_version": 1,
                    "source_breakdown": {
                        "frontier": {"candidate_connection_count": 1},
                        "watershed": {"candidate_connection_count": 0},
                        "fallback": {"candidate_connection_count": 0},
                    },
                }
            },
        },
        "network": {"matlab_count": 0, "python_count": 0, "matches_exactly": True},
    }


def test_full_comparison_managed_archive_compacts_and_refreshes_metadata(tmp_path, monkeypatch):
    import slavv.parity.comparison as comparison_module

    run_root = (
        tmp_path / "slavv_comparisons" / "experiments" / "live-parity" / "runs" / "20260418_live"
    )
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"tif")
    matlab_dir = run_root / "01_Input" / "matlab_results"
    python_dir = run_root / "02_Output" / "python_results"
    analysis_dir = run_root / "03_Analysis"
    metadata_dir = run_root / "99_Metadata"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = _write_matlab_archive_surface(matlab_dir)

    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(run_root),
    )
    status_reports = iter(
        [
            _make_status_report(matlab_dir, batch_dir=None),
            _make_status_report(matlab_dir, batch_dir=batch_dir),
        ]
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: next(status_reports),
    )
    monkeypatch.setattr(
        comparison_module,
        "run_matlab_vectorization",
        lambda *_args, **_kwargs: {
            "success": True,
            "elapsed_time": 0.0,
            "output_dir": str(matlab_dir),
            "batch_folder": str(batch_dir),
        },
    )
    monkeypatch.setattr(
        comparison_module,
        "run_python_vectorization",
        lambda *_args, **_kwargs: _write_python_archive_surface(python_dir),
    )
    monkeypatch.setattr(
        comparison_module,
        "load_matlab_batch_results",
        lambda _batch_folder: {
            "vertices": {"count": 1, "positions": []},
            "edges": {"count": 0, "connections": [], "traces": []},
            "network": {"strands": []},
            "network_stats": {"strand_count": 0},
        },
    )
    monkeypatch.setattr(
        comparison_module, "compare_results", lambda *_args, **_kwargs: _comparison_report()
    )
    monkeypatch.setattr(
        comparison_module,
        "build_shared_neighborhood_audit",
        lambda *_args, **_kwargs: {
            "top_neighborhood": {
                "origin_index": 1,
                "missing_matlab_incident_endpoint_pair_count": 1,
                "candidate_endpoint_pair_count": 1,
                "final_chosen_endpoint_pair_count": 1,
                "first_divergence_stage": "pre_manifest_rejection",
                "first_divergence_reason": "none",
            }
        },
    )

    result = orchestrate_comparison(
        input_file=str(input_file),
        output_dir=run_root,
        matlab_path="matlab.exe",
        project_root=tmp_path,
        params={"edge_method": "tracing"},
    )

    assert result == 0
    assert (metadata_dir / "status.json").exists()
    assert (metadata_dir / "artifact_cleanup.json").exists()
    assert (metadata_dir / "run_manifest.md").exists()
    assert (analysis_dir / "summary.txt").exists()
    assert (run_root / "02_Output" / "python_results" / "network.json").exists()
    assert (
        run_root / "02_Output" / "python_results" / "stages" / "edges" / "candidate_audit.json"
    ).exists()
    assert (
        run_root / "02_Output" / "python_results" / "stages" / "edges" / "candidate_lifecycle.json"
    ).exists()
    assert not (
        run_root / "02_Output" / "python_results" / "checkpoints" / "checkpoint_edges.pkl"
    ).exists()
    assert not (
        run_root / "02_Output" / "python_results" / "stages" / "edges" / "candidates.pkl"
    ).exists()
    assert not (run_root / "01_Input" / "matlab_results" / "matlab_run.log").exists()
    assert batch_dir.exists()
    assert (batch_dir / "data" / "energy_testROI").exists()
    assert (batch_dir / "vectors" / "curated_vertices_testROI.mat").exists()
    assert (batch_dir / "vectors" / "curated_edges_testROI.mat").exists()
    assert (batch_dir / "vectors" / "network_testROI.mat").exists()
    assert (batch_dir / "settings" / "batch.mat").exists()
    assert (batch_dir / "settings" / "energy_testROI.mat").exists()
    assert not (batch_dir / "vectors" / "preview_projection.tif").exists()
    assert (run_root.parent.parent / "index.json").exists()
    assert (run_root.parents[3] / "pointers" / "latest_completed.txt").read_text(
        encoding="utf-8"
    ).strip() == "experiments/live-parity/runs/20260418_live"


def test_standalone_comparison_managed_archive_keeps_analysis_surface(tmp_path, monkeypatch):
    import slavv.parity.comparison as comparison_module

    run_root = (
        tmp_path
        / "slavv_comparisons"
        / "experiments"
        / "standalone-check"
        / "runs"
        / "20260418_standalone"
    )
    matlab_dir = run_root / "01_Input" / "matlab_results"
    python_dir = run_root / "02_Output" / "python_results"
    analysis_dir = run_root / "03_Analysis"
    metadata_dir = run_root / "99_Metadata"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_matlab_archive_surface(matlab_dir)
    _write_python_archive_surface(python_dir)

    monkeypatch.setattr(
        comparison_module,
        "load_matlab_batch_results",
        lambda _batch_folder: {
            "vertices": {"count": 1, "positions": []},
            "edges": {"count": 0, "connections": [], "traces": []},
            "network": {"strands": []},
            "network_stats": {"strand_count": 0},
        },
    )
    monkeypatch.setattr(
        comparison_module, "compare_results", lambda *_args, **_kwargs: _comparison_report()
    )
    monkeypatch.setattr(
        comparison_module,
        "build_shared_neighborhood_audit",
        lambda *_args, **_kwargs: {
            "top_neighborhood": {
                "origin_index": 1,
                "missing_matlab_incident_endpoint_pair_count": 1,
                "candidate_endpoint_pair_count": 1,
                "final_chosen_endpoint_pair_count": 1,
                "first_divergence_stage": "pre_manifest_rejection",
                "first_divergence_reason": "none",
            }
        },
    )

    result = run_standalone_comparison(
        matlab_dir=matlab_dir,
        python_dir=python_dir,
        output_dir=run_root,
        project_root=tmp_path,
        python_result_source="network-json-only",
    )

    assert result == 0
    assert (metadata_dir / "artifact_cleanup.json").exists()
    assert (metadata_dir / "status.json").exists()
    assert (metadata_dir / "run_manifest.md").exists()
    assert (analysis_dir / "summary.txt").exists()
    assert (python_dir / "network.json").exists()
    assert (python_dir / "stages" / "edges" / "candidate_audit.json").exists()
    assert (python_dir / "stages" / "edges" / "candidate_lifecycle.json").exists()
    assert not (python_dir / "checkpoints" / "checkpoint_edges.pkl").exists()
    assert not (python_dir / "stages" / "edges" / "candidates.pkl").exists()
    batch_dir = matlab_dir / "batch_260418-120000"
    assert batch_dir.exists()
    assert (batch_dir / "data" / "energy_testROI").exists()
    assert (batch_dir / "vectors" / "network_testROI.mat").exists()
    assert (batch_dir / "settings" / "batch.mat").exists()
    assert not (batch_dir / "vectors" / "preview_projection.tif").exists()
