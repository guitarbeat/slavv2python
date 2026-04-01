"""Runtime-oriented helpers for MATLAB comparison execution."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from slavv.evaluation.comparison import (
    discover_matlab_artifacts,
    orchestrate_comparison,
    run_standalone_comparison,
)
from slavv.evaluation.management import generate_manifest
from slavv.evaluation.preflight import OutputRootPreflightReport
from slavv.runtime import RunContext, load_run_snapshot


def _make_preflight_report(
    output_root: Path,
    *,
    preflight_status: str = "passed",
    allows_launch: bool = True,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> OutputRootPreflightReport:
    return OutputRootPreflightReport(
        output_root=str(output_root),
        resolved_output_root=str(output_root.resolve()),
        preflight_status=preflight_status,
        allows_launch=allows_launch,
        writable=allows_launch,
        output_root_exists=True,
        output_root_created=False,
        free_space_gb=24.0,
        required_space_gb=5.0,
        onedrive_suspected=bool(warnings),
        warnings=list(warnings or []),
        errors=list(errors or []),
        recommended_action="Proceed." if allows_launch else "Choose a safer local output root.",
    )


def test_discover_matlab_artifacts_returns_empty_for_missing_output(tmp_path: Path):
    missing = tmp_path / "missing_output"

    assert discover_matlab_artifacts(missing) == {}


def test_discover_matlab_artifacts_prefers_latest_batch_and_network_file(tmp_path: Path):
    older = tmp_path / "batch_260323-180000"
    newer = tmp_path / "batch_260323-190000"
    (older / "vectors").mkdir(parents=True)
    newer_vectors = newer / "vectors"
    newer_vectors.mkdir(parents=True)
    (newer_vectors / "network_260323-190100_sample.mat").write_text("", encoding="utf-8")

    artifacts = discover_matlab_artifacts(tmp_path)

    assert artifacts["batch_folder"] == str(newer)
    assert artifacts["vectors_dir"] == str(newer_vectors)
    assert artifacts["network_mat"] == str(newer_vectors / "network_260323-190100_sample.mat")


def test_discover_matlab_artifacts_handles_partial_batch_without_network(tmp_path: Path):
    batch_folder = tmp_path / "batch_260323-200000"
    vectors_dir = batch_folder / "vectors"
    vectors_dir.mkdir(parents=True)

    artifacts = discover_matlab_artifacts(tmp_path)

    assert artifacts["batch_folder"] == str(batch_folder)
    assert artifacts["vectors_dir"] == str(vectors_dir)
    assert "network_mat" not in artifacts


def test_generate_manifest_includes_run_status(tmp_path: Path):
    run_dir = tmp_path / "comparison_run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
        provenance={"source": "comparison-test"},
    )
    context.mark_preprocess_complete()
    context.stage("energy").begin(detail="Energy resumed", units_total=4, units_completed=2)

    manifest = generate_manifest(run_dir)

    assert "## Run Status" in manifest
    assert "- **Status:** running" in manifest
    assert "- **Target stage:** network" in manifest


def test_orchestrate_comparison_updates_shared_run_snapshot(tmp_path: Path, monkeypatch):
    import slavv.evaluation.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"

    def fake_run_matlab_vectorization(
        _input, _output, _matlab_path, _project_root, params_file=None
    ):
        batch_folder.mkdir(parents=True, exist_ok=True)
        assert params_file is not None
        params_payload = json.loads(Path(params_file).read_text(encoding="utf-8"))
        assert params_payload["edge_method"] == "tracing"
        return {
            "success": True,
            "batch_folder": str(batch_folder),
            "elapsed_time": 12.0,
            "params_file": params_file,
        }

    def fake_run_python_vectorization(_input, output, _params, run_dir=None, force_rerun_from=None):
        Path(output).mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "elapsed_time": 3.0,
            "run_dir": run_dir,
            "force_rerun_from": force_rerun_from,
        }

    def fake_load_matlab_batch_results(_batch_folder):
        return {"timings": {"total": 12.0}}

    def fake_compare_results(_matlab_results, _python_results, matlab_parsed):
        assert matlab_parsed == {"timings": {"total": 12.0}}
        return {
            "matlab": {"elapsed_time": 12.0},
            "python": {"elapsed_time": 3.0},
            "performance": {"speedup": 4.0, "faster": "Python"},
            "vertices": {"matlab_count": 10, "python_count": 9},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"strand_delta": 0},
        }

    def fake_generate_summary(_output_dir, summary_file):
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text("summary", encoding="utf-8")

    def fake_generate_manifest(_output_dir, manifest_file):
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text("# manifest", encoding="utf-8")
        return "# manifest"

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fake_run_matlab_vectorization
    )
    monkeypatch.setattr(
        comparison_module, "run_python_vectorization", fake_run_python_vectorization
    )
    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_results", fake_load_matlab_batch_results
    )
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)
    monkeypatch.setattr(comparison_module, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(comparison_module, "generate_manifest", fake_generate_manifest)
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
    )

    snapshot = load_run_snapshot(output_dir)

    assert result == 0
    assert snapshot is not None
    assert snapshot.optional_tasks["output_preflight"].status == "completed"
    assert snapshot.optional_tasks["matlab_pipeline"].status == "completed"
    assert snapshot.optional_tasks["python_pipeline"].status == "completed"
    assert snapshot.optional_tasks["comparison_analysis"].status == "completed"
    assert snapshot.optional_tasks["manifest"].status == "completed"
    assert (
        snapshot.optional_tasks["matlab_pipeline"]
        .artifacts["params_file"]
        .endswith("comparison_params.normalized.json")
    )
    assert Path(
        snapshot.optional_tasks["comparison_analysis"].artifacts["comparison_report"]
    ).exists()
    assert (output_dir / "99_Metadata" / "output_preflight.json").exists()


def test_orchestrate_comparison_imports_matlab_energy_for_python_parity_run(
    tmp_path: Path, monkeypatch
):
    import slavv.evaluation.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"
    checkpoint_dir = output_dir / "02_Output" / "python_results" / "checkpoints"

    captured: dict[str, object] = {}

    def fake_run_matlab_vectorization(
        _input, _output, _matlab_path, _project_root, params_file=None
    ):
        batch_folder.mkdir(parents=True, exist_ok=True)
        assert params_file is not None
        return {
            "success": True,
            "batch_folder": str(batch_folder),
            "elapsed_time": 12.0,
            "params_file": params_file,
        }

    def fake_import_matlab_batch(batch, checkpoints, stages=None):
        captured["batch"] = batch
        captured["checkpoints"] = Path(checkpoints)
        captured["stages"] = stages
        Path(checkpoints).mkdir(parents=True, exist_ok=True)
        return {
            "energy": str(Path(checkpoints) / "checkpoint_energy.pkl"),
            "vertices": str(Path(checkpoints) / "checkpoint_vertices.pkl"),
        }

    def fake_load_matlab_batch_params(_batch_folder):
        return {
            "sigma_per_influence_vertices": 2.0,
            "sigma_per_influence_edges": 2.0 / 3.0,
        }

    def fake_run_python_vectorization(_input, output, _params, run_dir=None, force_rerun_from=None):
        captured["python_output"] = Path(output)
        captured["python_run_dir"] = run_dir
        captured["force_rerun_from"] = force_rerun_from
        captured["python_params"] = dict(_params)
        return {
            "success": True,
            "elapsed_time": 3.0,
            "run_dir": run_dir,
            "force_rerun_from": force_rerun_from,
        }

    def fake_load_matlab_batch_results(_batch_folder):
        return {"timings": {"total": 12.0}}

    def fake_compare_results(_matlab_results, _python_results, matlab_parsed):
        assert matlab_parsed == {"timings": {"total": 12.0}}
        return {
            "matlab": {"elapsed_time": 12.0},
            "python": {"elapsed_time": 3.0},
            "performance": {"speedup": 4.0, "faster": "Python"},
            "vertices": {"matlab_count": 10, "python_count": 9},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"strand_delta": 0},
        }

    def fake_generate_summary(_output_dir, summary_file):
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text("summary", encoding="utf-8")

    def fake_generate_manifest(_output_dir, manifest_file):
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text("# manifest", encoding="utf-8")
        return "# manifest"

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fake_run_matlab_vectorization
    )
    monkeypatch.setattr(comparison_module, "import_matlab_batch", fake_import_matlab_batch)
    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_params", fake_load_matlab_batch_params
    )
    monkeypatch.setattr(
        comparison_module, "run_python_vectorization", fake_run_python_vectorization
    )
    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_results", fake_load_matlab_batch_results
    )
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)
    monkeypatch.setattr(comparison_module, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(comparison_module, "generate_manifest", fake_generate_manifest)
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
    )

    snapshot = load_run_snapshot(output_dir)

    assert result == 0
    assert captured["batch"] == str(batch_folder)
    assert captured["checkpoints"] == checkpoint_dir
    assert captured["stages"] == ["energy", "vertices"]
    assert captured["python_output"] == output_dir / "02_Output" / "python_results"
    assert captured["python_run_dir"] == str(output_dir)
    assert captured["force_rerun_from"] == "edges"
    assert captured["python_params"]["sigma_per_influence_vertices"] == 2.0
    assert captured["python_params"]["sigma_per_influence_edges"] == 2.0 / 3.0
    assert snapshot is not None
    assert snapshot.optional_tasks["output_preflight"].status == "completed"
    assert snapshot.optional_tasks["matlab_import"].status == "completed"


def test_orchestrate_comparison_blocks_launch_on_fatal_preflight(
    tmp_path: Path, monkeypatch
):
    import slavv.evaluation.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    matlab_called = {"value": False}

    def fake_run_matlab_vectorization(*_args, **_kwargs):
        matlab_called["value"] = True
        return {"success": True}

    def fake_generate_manifest(_output_dir, manifest_file):
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text("# manifest", encoding="utf-8")
        return "# manifest"

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fake_run_matlab_vectorization
    )
    monkeypatch.setattr(comparison_module, "generate_manifest", fake_generate_manifest)
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(
            output_dir,
            preflight_status="blocked",
            allows_launch=False,
            errors=["Low disk space: 1.5 GB available (required minimum: 5.0 GB)"],
        ),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
    )

    snapshot = load_run_snapshot(output_dir)
    preflight_payload = json.loads(
        (output_dir / "99_Metadata" / "output_preflight.json").read_text(encoding="utf-8")
    )

    assert result == 1
    assert matlab_called["value"] is False
    assert snapshot is not None
    assert snapshot.status == "failed"
    assert snapshot.current_stage == "preflight"
    assert snapshot.optional_tasks["output_preflight"].status == "failed"
    assert snapshot.optional_tasks["manifest"].status == "completed"
    assert preflight_payload["preflight_status"] == "blocked"
    assert preflight_payload["allows_launch"] is False


def test_run_standalone_comparison_ignores_parameter_json(tmp_path: Path, monkeypatch):
    import slavv.evaluation.comparison as comparison_module

    run_dir = tmp_path / "comparison_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    batch_dir = matlab_dir / "batch_260323-190000"
    batch_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    (python_dir / "python_comparison_parameters.json").write_text("{}", encoding="utf-8")
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

    def fake_load_matlab_batch_results(_batch_folder):
        return {
            "vertices": {"count": 1, "positions": []},
            "edges": {"count": 0, "connections": [], "traces": []},
            "network": {"strands": []},
            "network_stats": {"strand_count": 0},
        }

    captured = {}

    def fake_compare_results(_matlab_results, python_results, _matlab_parsed):
        captured["python_results"] = python_results
        return {
            "matlab": {"elapsed_time": 0.0},
            "python": {"elapsed_time": 0.0},
            "performance": {},
            "vertices": {"matlab_count": 1, "python_count": 1},
            "edges": {"matlab_count": 0, "python_count": 0},
            "network": {"matlab_strand_count": 0, "python_strand_count": 0},
            "parity_gate": {"vertices_exact": True, "edges_exact": True, "strands_exact": True},
        }

    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_results", fake_load_matlab_batch_results
    )
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)

    result = run_standalone_comparison(matlab_dir, python_dir, run_dir, tmp_path)

    assert result == 0
    assert captured["python_results"]["vertices_count"] == 1
    assert "results" in captured["python_results"]


def test_run_standalone_comparison_uses_checkpoint_energy_source(tmp_path: Path, monkeypatch):
    import slavv.evaluation.comparison as comparison_module

    run_dir = tmp_path / "comparison_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    checkpoint_dir = python_dir / "checkpoints"
    stage_dir = python_dir / "stages" / "edges"
    batch_dir = matlab_dir / "batch_260326-090000"
    batch_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    stage_dir.mkdir(parents=True)

    joblib.dump(
        {"energy_origin": "matlab_batch_hdf5", "energy_source": "matlab_batch_hdf5"},
        checkpoint_dir / "checkpoint_energy.pkl",
    )
    joblib.dump({"positions": [[0, 0, 0]]}, checkpoint_dir / "checkpoint_vertices.pkl")
    joblib.dump({"connections": [[0, 0]], "traces": []}, checkpoint_dir / "checkpoint_edges.pkl")
    joblib.dump({"strands": []}, checkpoint_dir / "checkpoint_network.pkl")
    joblib.dump({"connections": [[0, 1], [1, 2]], "traces": []}, stage_dir / "candidates.pkl")

    def fake_load_matlab_batch_results(_batch_folder):
        return {
            "vertices": {"count": 1, "positions": []},
            "edges": {"count": 1, "connections": [[0, 0]], "traces": []},
            "network": {"strands": []},
            "network_stats": {"strand_count": 0},
        }

    captured = {}

    def fake_compare_results(_matlab_results, python_results, _matlab_parsed):
        captured["comparison_mode"] = python_results["comparison_mode"]
        captured["candidate_edges"] = python_results["results"].get("candidate_edges")
        return {
            "matlab": {"elapsed_time": 0.0},
            "python": {"elapsed_time": 0.0},
            "performance": {},
            "vertices": {"matlab_count": 1, "python_count": 1},
            "edges": {"matlab_count": 1, "python_count": 1},
            "network": {"matlab_strand_count": 0, "python_strand_count": 0},
            "parity_gate": {"vertices_exact": True, "edges_exact": True, "strands_exact": True},
        }

    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_results", fake_load_matlab_batch_results
    )
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)

    result = run_standalone_comparison(matlab_dir, python_dir, run_dir, tmp_path)

    assert result == 0
    assert captured["comparison_mode"]["result_source"] == "checkpoints"
    assert captured["comparison_mode"]["energy_source"] == "matlab_batch_hdf5"
    assert captured["candidate_edges"]["connections"] == [[0, 1], [1, 2]]
