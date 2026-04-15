"""Runtime-oriented helpers for MATLAB comparison execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest

from slavv.parity.comparison import (
    _format_progress_event_message,
    discover_matlab_artifacts,
    orchestrate_comparison,
    run_matlab_vectorization,
    run_standalone_comparison,
)
from slavv.parity.matlab_status import MatlabStatusReport
from slavv.parity.preflight import OutputRootPreflightReport
from slavv.parity.run_layout import generate_manifest
from slavv.runtime import ProgressEvent, RunContext, RunSnapshot, StageSnapshot, load_run_snapshot


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


def _make_matlab_status_report(
    output_root: Path,
    *,
    resume_mode: str = "fresh",
    batch_folder: Path | None = None,
    last_completed_stage: str = "",
    next_stage: str = "energy",
    rerun_prediction: str | None = None,
    failure_summary: str = "",
    stale_running_snapshot_suspected: bool = False,
    python_force_rerun_from: str | None = None,
) -> MatlabStatusReport:
    effective_batch_folder = batch_folder or (
        output_root / "01_Input" / "matlab_results" / "batch_260323-190000"
    )
    report = MatlabStatusReport(
        output_directory=str(output_root / "01_Input" / "matlab_results"),
        input_file=str(output_root / "input_volume.tif"),
        matlab_resume_state_file=str(
            output_root / "01_Input" / "matlab_results" / "matlab_resume_state.json"
        ),
        matlab_resume_state_present=True,
        matlab_resume_state_status="running:energy"
        if stale_running_snapshot_suspected
        else "completed",
        matlab_resume_state_updated_at="2026-04-01 12:27:34",
        matlab_log_file=str(output_root / "01_Input" / "matlab_results" / "matlab_run.log"),
        matlab_log_present=True,
        matlab_batch_folder=str(effective_batch_folder) if batch_folder is not None else "",
        matlab_batch_timestamp=effective_batch_folder.name.replace("batch_", "")
        if batch_folder is not None
        else "",
        matlab_batch_complete=resume_mode == "complete-noop",
        matlab_resume_mode=resume_mode,
        matlab_last_completed_stage=last_completed_stage,
        matlab_next_stage=next_stage,
        matlab_partial_stage_artifacts_present=resume_mode == "restart-current-stage",
        matlab_partial_stage_name=next_stage if resume_mode == "restart-current-stage" else "",
        matlab_rerun_prediction=rerun_prediction
        or f"Rerun will reuse {effective_batch_folder.name} and start at {next_stage}.",
        stale_running_snapshot_suspected=stale_running_snapshot_suspected,
        failure_summary=failure_summary,
        matlab_log_tail=[failure_summary] if failure_summary else [],
        authoritative_files={
            "resume_state": str(
                output_root / "01_Input" / "matlab_results" / "matlab_resume_state.json"
            ),
            "matlab_log": str(output_root / "01_Input" / "matlab_results" / "matlab_run.log"),
        },
    )
    if python_force_rerun_from is not None:
        report.authoritative_files["python_force_rerun_from"] = python_force_rerun_from
    return report


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


def test_format_progress_event_message_explains_energy_stage_units():
    snapshot = RunSnapshot(
        run_id="run-1",
        overall_progress=0.14,
        stages={
            "energy": StageSnapshot(
                name="energy",
                status="running",
                progress=0.2654,
                detail="Energy volume tile 136/512, vessel scale 23/26",
                substage="scale_chunks",
                units_completed=3533,
                units_total=13312,
                eta_seconds=2596.0,
            )
        },
    )
    event = ProgressEvent(
        stage="energy",
        status="running",
        overall_progress=snapshot.overall_progress,
        stage_progress=snapshot.stages["energy"].progress,
        detail=snapshot.stages["energy"].detail,
        resumed=False,
        snapshot=snapshot,
    )

    message = _format_progress_event_message(event)

    assert "Energy 26.5%" in message
    assert "Energy volume tile 136/512, vessel scale 23/26" in message
    assert "3,533/13,312 work units" in message
    assert "ETA" in message


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher only runs on Windows")
def test_run_matlab_vectorization_launches_batch_wrapper_via_cmd(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

    repo_root = Path(__file__).resolve().parents[4]
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
    output_dir = tmp_path / "matlab_results"
    params_file = tmp_path / "comparison params.json"
    params_file.write_text("{}", encoding="utf-8")
    mock_matlab = tmp_path / "mock_matlab.bat"
    mock_matlab.write_text(
        '@echo off\necho "MOCK MATLAB CALLED WITH: %*"\nexit /b 0',
        encoding="utf-8",
    )

    monkeypatch.setattr(comparison_module, "get_system_info", lambda: {"platform": "test"})
    monkeypatch.setattr(
        comparison_module,
        "get_matlab_info",
        lambda _path: {"version": "mock", "available": True},
    )

    result = run_matlab_vectorization(
        str(input_file),
        str(output_dir),
        str(mock_matlab),
        repo_root,
        batch_script=str(repo_root / "dev" / "scripts" / "cli" / "run_matlab_cli.bat"),
        params_file=str(params_file),
    )

    log_file = output_dir / "matlab_run.log"
    assert result["success"] is True
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "MOCK MATLAB CALLED WITH:" in content
    assert "-wait -batch" in content


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
    import slavv.parity.comparison as comparison_module

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
        assert params_payload["comparison_exact_network"] is True
        assert params_payload["python_parity_rerun_from"] == "edges"
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
    matlab_status_reports = iter(
        [
            _make_matlab_status_report(
                output_dir,
                resume_mode="fresh",
                batch_folder=None,
                rerun_prediction="No reusable MATLAB batch found; rerun will create a new batch and start at energy.",
            ),
            _make_matlab_status_report(
                output_dir,
                resume_mode="complete-noop",
                batch_folder=batch_folder,
                last_completed_stage="network",
                next_stage="",
                rerun_prediction="batch_260323-190000 is already complete; rerun should be a no-op unless inputs change.",
            ),
        ]
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: next(matlab_status_reports),
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
    assert snapshot.optional_tasks["matlab_status"].status == "completed"
    assert snapshot.optional_tasks["matlab_pipeline"].status == "completed"
    assert snapshot.optional_tasks["python_pipeline"].status == "completed"
    assert snapshot.optional_tasks["comparison_analysis"].status == "completed"
    assert snapshot.optional_tasks["manifest"].status == "completed"
    assert snapshot.optional_tasks["matlab_status"].artifacts["resume_mode"] == "complete-noop"
    assert (
        snapshot.optional_tasks["matlab_pipeline"]
        .artifacts["params_file"]
        .endswith("comparison_params.normalized.json")
    )
    assert Path(
        snapshot.optional_tasks["comparison_analysis"].artifacts["comparison_report"]
    ).exists()
    assert (output_dir / "99_Metadata" / "output_preflight.json").exists()


def test_orchestrate_comparison_persists_shared_neighborhood_diagnostics(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"

    monkeypatch.setattr(
        comparison_module,
        "run_matlab_vectorization",
        lambda *_args, **_kwargs: {
            "success": True,
            "batch_folder": str(batch_folder),
            "elapsed_time": 12.0,
        },
    )
    monkeypatch.setattr(
        comparison_module,
        "run_python_vectorization",
        lambda *_args, **_kwargs: {
            "success": True,
            "elapsed_time": 3.0,
            "results": {},
        },
    )
    monkeypatch.setattr(
        comparison_module,
        "load_matlab_batch_results",
        lambda _batch_folder: {"timings": {"total": 12.0}},
    )
    monkeypatch.setattr(
        comparison_module,
        "generate_summary",
        lambda _output_dir, summary_file: summary_file.write_text("summary", encoding="utf-8"),
    )
    monkeypatch.setattr(
        comparison_module,
        "generate_manifest",
        lambda _output_dir, manifest_file: manifest_file.write_text("# manifest", encoding="utf-8"),
    )
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: _make_matlab_status_report(
            output_dir,
            resume_mode="complete-noop",
            batch_folder=batch_folder,
            last_completed_stage="network",
            next_stage="",
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "compare_results",
        lambda *_args, **_kwargs: {
            "matlab": {"elapsed_time": 12.0},
            "python": {"elapsed_time": 3.0},
            "performance": {"speedup": 4.0, "faster": "Python"},
            "edges": {
                "matlab_count": 5,
                "python_count": 3,
                "exact_match": False,
                "diagnostics": {
                    "candidate_endpoint_coverage": {
                        "matlab_endpoint_pair_count": 5,
                        "matched_matlab_endpoint_pair_count": 2,
                        "missing_matlab_endpoint_pair_count": 3,
                        "candidate_endpoint_pair_count": 4,
                        "python_endpoint_pair_count": 3,
                        "extra_candidate_endpoint_pair_count": 1,
                    },
                    "shared_neighborhood_audit": {
                        "neighborhoods": [
                            {
                                "origin_index": 866,
                                "selection_sources": ["tracked_hotspot"],
                                "matlab_incident_endpoint_pair_count": 4,
                                "candidate_endpoint_pair_count": 1,
                                "final_chosen_endpoint_pair_count": 1,
                                "missing_matlab_incident_endpoint_pair_count": 3,
                                "extra_candidate_endpoint_pair_count": 0,
                                "missing_final_endpoint_pair_count": 3,
                                "missing_matlab_incident_endpoint_pair_samples": [[866, 10]],
                                "candidate_endpoint_pair_samples": [[866, 22]],
                                "first_divergence_stage": "pre_manifest_rejection",
                                "first_divergence_reason": "rejected_parent_has_child",
                            }
                        ]
                    },
                },
            },
        },
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
    )

    assert result == 0
    assert (output_dir / "03_Analysis" / "shared_neighborhood_diagnostics.json").exists()
    assert (output_dir / "03_Analysis" / "shared_neighborhood_diagnostics.md").exists()


def test_orchestrate_comparison_full_reuse_analysis_only_skips_matlab_launch(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"
    batch_folder.mkdir(parents=True)
    metadata_dir = output_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "matlab_status.json").write_text(
        json.dumps(
            {
                "matlab_resume_mode": "complete-noop",
                "matlab_batch_complete": True,
                "matlab_batch_folder": str(batch_folder),
            }
        ),
        encoding="utf-8",
    )
    python_output = output_dir / "02_Output" / "python_results"
    python_output.mkdir(parents=True)
    (python_output / "network.json").write_text("{}", encoding="utf-8")

    def fail_run_matlab_vectorization(*_args, **_kwargs):
        raise AssertionError("MATLAB should not be launched when a completed reusable batch exists")

    def fail_run_python_vectorization(*_args, **_kwargs):
        raise AssertionError("Python should not rerun in analysis-only reuse mode")

    def fail_import_matlab_batch(*_args, **_kwargs):
        raise AssertionError("MATLAB import should not run in analysis-only reuse mode")

    def fake_load_matlab_batch_results(_batch_folder):
        return {"timings": {"total": 12.0}}

    def fake_compare_results(_matlab_results, python_results, matlab_parsed):
        assert python_results.get("success") is True
        assert matlab_parsed == {"timings": {"total": 12.0}}
        return {
            "matlab": {"elapsed_time": 0.0},
            "python": {"elapsed_time": 0.0},
            "performance": {"speedup": 1.0, "faster": "equal"},
            "vertices": {"matlab_count": 10, "python_count": 10},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"strand_delta": 0},
        }

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fail_run_matlab_vectorization
    )
    monkeypatch.setattr(
        comparison_module, "run_python_vectorization", fail_run_python_vectorization
    )
    monkeypatch.setattr(comparison_module, "import_matlab_batch", fail_import_matlab_batch)
    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_results", fake_load_matlab_batch_results
    )
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)
    monkeypatch.setattr(
        comparison_module,
        "generate_summary",
        lambda _output_dir, summary_file: (
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            or summary_file.write_text("summary", encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "generate_manifest",
        lambda _output_dir, manifest_file: (
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            or manifest_file.write_text("# manifest", encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: _make_matlab_status_report(
            output_dir,
            resume_mode="complete-noop",
            batch_folder=batch_folder,
            last_completed_stage="network",
            next_stage="",
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

    assert result == 0
    assert snapshot is not None
    assert snapshot.optional_tasks["matlab_pipeline"].artifacts["launch"] == "skipped"
    assert snapshot.optional_tasks["matlab_pipeline"].artifacts["reuse_mode"] == "analysis-only"
    assert (
        snapshot.optional_tasks["matlab_status"].artifacts["matlab_launch_skip_reason"]
        == "completed_reusable_batch"
    )
    assert snapshot.optional_tasks["comparison_analysis"].status == "completed"
    assert Path(
        snapshot.optional_tasks["comparison_analysis"].artifacts["comparison_report"]
    ).exists()


def test_orchestrate_comparison_full_reuse_python_rerun_skips_matlab_launch(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"
    batch_folder.mkdir(parents=True)
    metadata_dir = output_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "matlab_status.json").write_text(
        json.dumps(
            {
                "matlab_resume_mode": "complete-noop",
                "matlab_batch_complete": True,
                "matlab_batch_folder": str(batch_folder),
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fail_run_matlab_vectorization(*_args, **_kwargs):
        raise AssertionError("MATLAB should not be relaunched for completed reusable batches")

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
        return {"sigma_per_influence_vertices": 2.0}

    def fake_run_python_vectorization(_input, output, _params, run_dir=None, force_rerun_from=None):
        captured["python_output"] = Path(output)
        captured["python_run_dir"] = run_dir
        captured["force_rerun_from"] = force_rerun_from
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
            "matlab": {"elapsed_time": 0.0},
            "python": {"elapsed_time": 3.0},
            "performance": {"speedup": 0.0, "faster": "MATLAB"},
            "vertices": {"matlab_count": 10, "python_count": 9},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"strand_delta": 0},
        }

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fail_run_matlab_vectorization
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
    monkeypatch.setattr(
        comparison_module,
        "generate_summary",
        lambda _output_dir, summary_file: (
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            or summary_file.write_text("summary", encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "generate_manifest",
        lambda _output_dir, manifest_file: (
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            or manifest_file.write_text("# manifest", encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: _make_matlab_status_report(
            output_dir,
            resume_mode="complete-noop",
            batch_folder=batch_folder,
            last_completed_stage="network",
            next_stage="",
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

    assert result == 0
    assert captured["batch"] == str(batch_folder)
    assert captured["stages"] == ["energy", "vertices"]
    assert captured["force_rerun_from"] == "edges"
    assert snapshot is not None
    assert snapshot.optional_tasks["matlab_pipeline"].artifacts["launch"] == "skipped"
    assert snapshot.optional_tasks["matlab_pipeline"].artifacts["reuse_mode"] == "python-rerun"
    assert snapshot.optional_tasks["matlab_import"].status == "completed"


def test_orchestrate_comparison_shallow_mode_skips_matlab_parse(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

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

    def fail_load_matlab_batch_results(_batch_folder):
        raise AssertionError("deep MATLAB parsing should be skipped in shallow mode")

    captured = {}

    def fake_compare_results(_matlab_results, _python_results, matlab_parsed):
        captured["matlab_parsed"] = matlab_parsed
        return {
            "matlab": {"elapsed_time": 12.0},
            "python": {"elapsed_time": 3.0},
            "performance": {"speedup": 4.0, "faster": "Python"},
            "vertices": {"matlab_count": 10, "python_count": 9},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"strand_delta": 0},
        }

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fake_run_matlab_vectorization
    )
    monkeypatch.setattr(
        comparison_module, "run_python_vectorization", fake_run_python_vectorization
    )
    monkeypatch.setattr(
        comparison_module, "load_matlab_batch_results", fail_load_matlab_batch_results
    )
    monkeypatch.setattr(comparison_module, "compare_results", fake_compare_results)
    monkeypatch.setattr(
        comparison_module,
        "generate_summary",
        lambda _output_dir, summary_file: (
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            or summary_file.write_text("summary", encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "generate_manifest",
        lambda _output_dir, manifest_file: (
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            or manifest_file.write_text("# manifest", encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )
    matlab_status_reports = iter(
        [
            _make_matlab_status_report(output_dir, resume_mode="fresh", batch_folder=None),
            _make_matlab_status_report(
                output_dir,
                resume_mode="complete-noop",
                batch_folder=batch_folder,
                last_completed_stage="network",
                next_stage="",
            ),
        ]
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: next(matlab_status_reports),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
        comparison_depth="shallow",
    )

    assert result == 0
    assert captured["matlab_parsed"] is None


def test_orchestrate_comparison_imports_matlab_energy_for_python_parity_run(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

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
    matlab_status_reports = iter(
        [
            _make_matlab_status_report(
                output_dir,
                resume_mode="fresh",
                batch_folder=None,
                rerun_prediction="No reusable MATLAB batch found; rerun will create a new batch and start at energy.",
            ),
            _make_matlab_status_report(
                output_dir,
                resume_mode="resume-stage",
                batch_folder=batch_folder,
                last_completed_stage="vertices",
                next_stage="edges",
                rerun_prediction="Rerun will reuse batch_260323-190000 and start at edges.",
            ),
        ]
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: next(matlab_status_reports),
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
    assert captured["python_params"]["comparison_exact_network"] is True
    assert captured["python_params"]["python_parity_rerun_from"] == "edges"
    assert captured["python_params"]["sigma_per_influence_vertices"] == 2.0
    assert captured["python_params"]["sigma_per_influence_edges"] == 2.0 / 3.0
    assert snapshot is not None
    assert snapshot.optional_tasks["output_preflight"].status == "completed"
    assert snapshot.optional_tasks["matlab_status"].artifacts["python_force_rerun_from"] == "edges"
    assert snapshot.optional_tasks["matlab_import"].status == "completed"


def test_orchestrate_comparison_skip_matlab_reuses_existing_batch_for_python_parity_run(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"
    checkpoint_dir = output_dir / "02_Output" / "python_results" / "checkpoints"

    captured: dict[str, object] = {}

    def fail_run_matlab_vectorization(*_args, **_kwargs):
        raise AssertionError("MATLAB should not be launched when --skip-matlab is set")

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
        return {"sigma_per_influence_vertices": 2.0}

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
        captured["parsed_batch"] = _batch_folder
        return {"timings": {"total": 12.0}}

    def fake_compare_results(matlab_results, _python_results, matlab_parsed):
        captured["matlab_results"] = dict(matlab_results)
        assert matlab_parsed == {"timings": {"total": 12.0}}
        return {
            "matlab": {"elapsed_time": 0.0},
            "python": {"elapsed_time": 3.0},
            "performance": {},
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
        comparison_module, "run_matlab_vectorization", fail_run_matlab_vectorization
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
        "inspect_matlab_status",
        lambda *_args, **_kwargs: _make_matlab_status_report(
            output_dir,
            resume_mode="complete-noop",
            batch_folder=batch_folder,
            last_completed_stage="network",
            next_stage="",
            rerun_prediction="batch_260323-190000 is already complete; reuse it for Python parity work.",
        ),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
        skip_matlab=True,
    )

    snapshot = load_run_snapshot(output_dir)

    assert result == 0
    assert captured["batch"] == str(batch_folder)
    assert captured["checkpoints"] == checkpoint_dir
    assert captured["stages"] == ["energy", "vertices"]
    assert captured["python_output"] == output_dir / "02_Output" / "python_results"
    assert captured["python_run_dir"] == str(output_dir)
    assert captured["force_rerun_from"] == "edges"
    assert captured["python_params"]["comparison_exact_network"] is True
    assert captured["python_params"]["python_parity_rerun_from"] == "edges"
    assert captured["python_params"]["sigma_per_influence_vertices"] == 2.0
    assert captured["matlab_results"]["batch_folder"] == str(batch_folder)
    assert captured["parsed_batch"] == str(batch_folder)
    assert snapshot is not None
    assert snapshot.optional_tasks["matlab_status"].artifacts["python_force_rerun_from"] == "edges"
    assert snapshot.optional_tasks["matlab_import"].status == "completed"


def test_orchestrate_comparison_can_rerun_python_from_network_with_imported_matlab_edges(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

    input_file = tmp_path / "input_volume.tif"
    input_file.write_bytes(b"fake-tiff")
    output_dir = tmp_path / "comparison_run"
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    batch_folder = output_dir / "01_Input" / "matlab_results" / "batch_260323-190000"
    checkpoint_dir = output_dir / "02_Output" / "python_results" / "checkpoints"

    captured: dict[str, object] = {}

    def fail_run_matlab_vectorization(*_args, **_kwargs):
        raise AssertionError("MATLAB should not be launched when --skip-matlab is set")

    def fake_import_matlab_batch(batch, checkpoints, stages=None):
        captured["batch"] = batch
        captured["checkpoints"] = Path(checkpoints)
        captured["stages"] = stages
        Path(checkpoints).mkdir(parents=True, exist_ok=True)
        return {
            "energy": str(Path(checkpoints) / "checkpoint_energy.pkl"),
            "vertices": str(Path(checkpoints) / "checkpoint_vertices.pkl"),
            "edges": str(Path(checkpoints) / "checkpoint_edges.pkl"),
        }

    def fake_load_matlab_batch_params(_batch_folder):
        return {"sigma_per_influence_vertices": 2.0}

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
        captured["parsed_batch"] = _batch_folder
        return {"timings": {"total": 12.0}}

    def fake_compare_results(matlab_results, _python_results, matlab_parsed):
        captured["matlab_results"] = dict(matlab_results)
        assert matlab_parsed == {"timings": {"total": 12.0}}
        return {
            "matlab": {"elapsed_time": 0.0},
            "python": {"elapsed_time": 3.0},
            "performance": {},
            "vertices": {"matlab_count": 10, "python_count": 10},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"exact_match": True},
        }

    def fake_generate_summary(_output_dir, summary_file):
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text("summary", encoding="utf-8")

    def fake_generate_manifest(_output_dir, manifest_file):
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text("# manifest", encoding="utf-8")
        return "# manifest"

    monkeypatch.setattr(
        comparison_module, "run_matlab_vectorization", fail_run_matlab_vectorization
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
        "inspect_matlab_status",
        lambda *_args, **_kwargs: _make_matlab_status_report(
            output_dir,
            resume_mode="complete-noop",
            batch_folder=batch_folder,
            last_completed_stage="network",
            next_stage="",
            rerun_prediction="batch_260323-190000 is already complete; reuse it for Python parity work.",
        ),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing", "comparison_exact_network": False},
        skip_matlab=True,
        python_parity_rerun_from="network",
    )

    snapshot = load_run_snapshot(output_dir)

    assert result == 0
    assert captured["batch"] == str(batch_folder)
    assert captured["checkpoints"] == checkpoint_dir
    assert captured["stages"] == ["energy", "vertices", "edges"]
    assert captured["python_output"] == output_dir / "02_Output" / "python_results"
    assert captured["python_run_dir"] == str(output_dir)
    assert captured["force_rerun_from"] == "network"
    assert captured["python_params"]["comparison_exact_network"] is True
    assert captured["python_params"]["python_parity_rerun_from"] == "network"
    assert captured["python_params"]["sigma_per_influence_vertices"] == 2.0
    assert captured["matlab_results"]["batch_folder"] == str(batch_folder)
    assert captured["parsed_batch"] == str(batch_folder)
    assert snapshot is not None
    assert (
        snapshot.optional_tasks["matlab_status"].artifacts["python_force_rerun_from"] == "network"
    )
    assert snapshot.optional_tasks["matlab_import"].status == "completed"
    assert (
        snapshot.optional_tasks["matlab_import"].artifacts["python_parity_rerun_from"] == "network"
    )


def test_orchestrate_comparison_imports_matlab_edges_for_fresh_network_stage_probe(
    tmp_path: Path, monkeypatch
):
    import slavv.parity.comparison as comparison_module

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
        params_payload = json.loads(Path(params_file).read_text(encoding="utf-8"))
        assert params_payload["comparison_exact_network"] is True
        assert params_payload["python_parity_rerun_from"] == "network"
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
            "edges": str(Path(checkpoints) / "checkpoint_edges.pkl"),
        }

    def fake_load_matlab_batch_params(_batch_folder):
        return {"sigma_per_influence_vertices": 2.0}

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
            "vertices": {"matlab_count": 10, "python_count": 10},
            "edges": {"matlab_count": 5, "python_count": 5},
            "network": {"exact_match": True},
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
    matlab_status_reports = iter(
        [
            _make_matlab_status_report(
                output_dir,
                resume_mode="fresh",
                batch_folder=None,
                rerun_prediction="No reusable MATLAB batch found; rerun will create a new batch and start at energy.",
            ),
            _make_matlab_status_report(
                output_dir,
                resume_mode="complete-noop",
                batch_folder=batch_folder,
                last_completed_stage="network",
                next_stage="",
                rerun_prediction="batch_260323-190000 is already complete; rerun should be a no-op unless inputs change.",
            ),
        ]
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: next(matlab_status_reports),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
        python_parity_rerun_from="network",
    )

    snapshot = load_run_snapshot(output_dir)

    assert result == 0
    assert captured["batch"] == str(batch_folder)
    assert captured["checkpoints"] == checkpoint_dir
    assert captured["stages"] == ["energy", "vertices", "edges"]
    assert captured["python_output"] == output_dir / "02_Output" / "python_results"
    assert captured["python_run_dir"] == str(output_dir)
    assert captured["force_rerun_from"] == "network"
    assert captured["python_params"]["comparison_exact_network"] is True
    assert captured["python_params"]["python_parity_rerun_from"] == "network"
    assert snapshot is not None
    assert (
        snapshot.optional_tasks["matlab_status"].artifacts["python_force_rerun_from"] == "network"
    )
    assert (
        snapshot.optional_tasks["matlab_import"].artifacts["python_parity_rerun_from"] == "network"
    )


def test_orchestrate_comparison_blocks_launch_on_fatal_preflight(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

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


def test_orchestrate_comparison_persists_matlab_failure_summary(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

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
        return {
            "success": False,
            "batch_folder": str(batch_folder),
            "elapsed_time": 12.0,
            "params_file": params_file,
            "error": "MATLAB exited with code 1",
            "log_file": str(output_dir / "01_Input" / "matlab_results" / "matlab_run.log"),
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
    monkeypatch.setattr(comparison_module, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(comparison_module, "generate_manifest", fake_generate_manifest)
    monkeypatch.setattr(
        comparison_module,
        "evaluate_output_root_preflight",
        lambda _output_root: _make_preflight_report(output_dir),
    )
    matlab_status_reports = iter(
        [
            _make_matlab_status_report(
                output_dir,
                resume_mode="fresh",
                batch_folder=None,
                rerun_prediction="No reusable MATLAB batch found; rerun will create a new batch and start at energy.",
            ),
            _make_matlab_status_report(
                output_dir,
                resume_mode="restart-current-stage",
                batch_folder=batch_folder,
                next_stage="energy",
                rerun_prediction="Rerun will reuse batch_260323-190000 but restart energy from the stage boundary. Partial energy artifacts were found.",
                failure_summary="ERROR: MATLAB error Exit Status: 0x00000001",
                stale_running_snapshot_suspected=True,
            ),
        ]
    )
    monkeypatch.setattr(
        comparison_module,
        "inspect_matlab_status",
        lambda *_args, **_kwargs: next(matlab_status_reports),
    )

    result = orchestrate_comparison(
        str(input_file),
        output_dir,
        "matlab.exe",
        project_root,
        params={"edge_method": "tracing"},
        skip_python=True,
    )

    snapshot = load_run_snapshot(output_dir)
    failure_summary_file = output_dir / "99_Metadata" / "matlab_failure_summary.json"
    matlab_status_file = output_dir / "99_Metadata" / "matlab_status.json"
    failure_summary = json.loads(failure_summary_file.read_text(encoding="utf-8"))

    assert result == 1
    assert snapshot is not None
    assert snapshot.status == "failed"
    assert snapshot.current_stage == "matlab"
    assert snapshot.optional_tasks["matlab_status"].status == "failed"
    assert snapshot.optional_tasks["matlab_pipeline"].status == "failed"
    assert failure_summary_file.exists()
    assert matlab_status_file.exists()
    assert failure_summary["failure_summary"] == "ERROR: MATLAB error Exit Status: 0x00000001"


def test_run_standalone_comparison_ignores_parameter_json(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

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
    import slavv.parity.comparison as comparison_module

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


def test_run_standalone_comparison_can_force_network_json_source(tmp_path: Path, monkeypatch):
    import slavv.parity.comparison as comparison_module

    run_dir = tmp_path / "comparison_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    checkpoint_dir = python_dir / "checkpoints"
    batch_dir = matlab_dir / "batch_260326-090000"
    batch_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    joblib.dump({"positions": [[9, 9, 9]]}, checkpoint_dir / "checkpoint_vertices.pkl")
    (python_dir / "network.json").write_text(
        json.dumps(
            {
                "vertices": {"positions": [[1, 2, 3]]},
                "edges": {"connections": [[0, 0]], "traces": []},
                "network": {"strands": []},
            }
        ),
        encoding="utf-8",
    )

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
        captured["vertex_positions"] = python_results["results"]["vertices"]["positions"].tolist()
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

    result = run_standalone_comparison(
        matlab_dir,
        python_dir,
        run_dir,
        tmp_path,
        python_result_source="network-json-only",
    )

    assert result == 0
    assert captured["comparison_mode"]["result_source"] == "network_json"
    assert captured["vertex_positions"] == [[1, 2, 3]]
