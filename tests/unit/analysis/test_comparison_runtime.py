"""Runtime-oriented helpers for MATLAB comparison execution."""

from __future__ import annotations

from pathlib import Path

from slavv.evaluation.comparison import discover_matlab_artifacts, orchestrate_comparison
from slavv.evaluation.management import generate_manifest
from slavv.runtime import RunContext, load_run_snapshot


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

    def fake_run_matlab_vectorization(_input, _output, _matlab_path, _project_root):
        batch_folder.mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "batch_folder": str(batch_folder),
            "elapsed_time": 12.0,
        }

    def fake_run_python_vectorization(_input, output, _params, run_dir=None):
        Path(output).mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "elapsed_time": 3.0,
            "run_dir": run_dir,
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
    assert snapshot.optional_tasks["matlab_pipeline"].status == "completed"
    assert snapshot.optional_tasks["python_pipeline"].status == "completed"
    assert snapshot.optional_tasks["comparison_analysis"].status == "completed"
    assert snapshot.optional_tasks["manifest"].status == "completed"
    assert Path(
        snapshot.optional_tasks["comparison_analysis"].artifacts["comparison_report"]
    ).exists()
