from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from slavv.parity.matlab_status import MatlabStatusReport, persist_matlab_status
from slavv.parity.preflight import OutputRootPreflightReport, persist_output_preflight
from slavv.parity.workflow_assessment import (
    assess_loop_request,
    evaluate_output_root_preflight_cached,
    inspect_matlab_status_cached,
)

if TYPE_CHECKING:
    from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def test_assess_loop_request_marks_skip_matlab_root_reusable_with_checkpoints(tmp_path: Path):
    run_root = tmp_path / "20260410_120000_comparison"
    metadata_dir = run_root / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(tmp_path / "input.tif")}}),
        encoding="utf-8",
    )
    (metadata_dir / "comparison_params.normalized.json").write_text(
        json.dumps(
            {
                "edge_method": "tracing",
                "comparison_exact_network": True,
                "python_parity_rerun_from": "edges",
            }
        ),
        encoding="utf-8",
    )
    checkpoints_dir = run_root / "02_Output" / "python_results" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "checkpoint_edges.pkl").write_bytes(b"checkpoint")

    report = assess_loop_request(
        run_root,
        loop_kind="skip_matlab_edges",
        input_path=tmp_path / "input.tif",
        params={
            "edge_method": "tracing",
            "comparison_exact_network": True,
            "python_parity_rerun_from": "edges",
        },
    )

    assert report.verdict == "reuse_ready"
    assert report.safe_to_reuse is True
    assert report.requires_fresh_matlab is True


def test_evaluate_output_root_preflight_cached_reuses_recent_report(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "comparison_root"
    output_root.mkdir()
    metadata_dir = output_root / "99_Metadata"
    report = OutputRootPreflightReport(
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
        cache_created_at=_utc_now_iso(),
        cache_valid_for_seconds=300,
    )
    persist_output_preflight(report, metadata_dir)
    monkeypatch.setattr(
        "slavv.parity.workflow_assessment.evaluate_output_root_preflight",
        lambda *_args, **_kwargs: iter(()).throw(
            AssertionError("cache should be reused")
        ),
    )

    cached = evaluate_output_root_preflight_cached(output_root, metadata_dir)

    assert cached.cache_used is True
    assert cached.preflight_status == "passed"


def test_inspect_matlab_status_cached_reuses_when_mtimes_match(tmp_path: Path, monkeypatch):
    matlab_output = tmp_path / "matlab_results"
    matlab_output.mkdir()
    resume_state = matlab_output / "matlab_resume_state.json"
    resume_state.write_text("{}", encoding="utf-8")
    matlab_log = matlab_output / "matlab_run.log"
    matlab_log.write_text("ok", encoding="utf-8")
    batch_folder = matlab_output / "batch_260410-120000"
    batch_folder.mkdir()
    metadata_dir = tmp_path / "99_Metadata"
    report = MatlabStatusReport(
        output_directory=str(matlab_output),
        input_file=str(tmp_path / "input.tif"),
        matlab_resume_state_file=str(resume_state),
        matlab_log_file=str(matlab_log),
        matlab_batch_folder=str(batch_folder),
        matlab_resume_mode="fresh",
        matlab_next_stage="energy",
        matlab_rerun_prediction="Rerun will start at energy.",
        cache_created_at=_utc_now_iso(),
        matlab_resume_state_mtime=resume_state.stat().st_mtime,
        matlab_log_mtime=matlab_log.stat().st_mtime,
        matlab_batch_folder_mtime=batch_folder.stat().st_mtime,
    )
    persist_matlab_status(report, metadata_dir)
    monkeypatch.setattr(
        "slavv.parity.workflow_assessment.inspect_matlab_status",
        lambda *_args, **_kwargs: iter(()).throw(
            AssertionError("cache should be reused")
        ),
    )

    cached = inspect_matlab_status_cached(
        matlab_output,
        metadata_dir,
        input_file=tmp_path / "input.tif",
    )

    assert cached.cache_used is True
    assert cached.matlab_rerun_prediction == "Rerun will start at energy."


def test_assess_full_comparison_analysis_ready_with_completed_matlab_and_python_results(
    tmp_path: Path,
):
    run_root = tmp_path / "20260413_120000_comparison"
    metadata_dir = run_root / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(tmp_path / "input.tif")}}),
        encoding="utf-8",
    )
    (metadata_dir / "comparison_params.normalized.json").write_text(
        json.dumps(
            {
                "edge_method": "tracing",
                "comparison_exact_network": True,
                "python_parity_rerun_from": "edges",
            }
        ),
        encoding="utf-8",
    )
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260413-120000"
    batch_dir.mkdir(parents=True)
    (metadata_dir / "matlab_status.json").write_text(
        json.dumps(
            {
                "matlab_resume_mode": "complete-noop",
                "matlab_batch_complete": True,
                "matlab_batch_folder": str(batch_dir),
            }
        ),
        encoding="utf-8",
    )
    python_dir = run_root / "02_Output" / "python_results"
    python_dir.mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")

    report = assess_loop_request(
        run_root,
        loop_kind="full_comparison",
        input_path=tmp_path / "input.tif",
        params={
            "edge_method": "tracing",
            "comparison_exact_network": True,
            "python_parity_rerun_from": "edges",
        },
    )

    assert report.verdict == "analysis_ready"
    assert report.safe_to_reuse is True
    assert report.safe_to_analyze_only is True
    assert report.requires_fresh_matlab is False


def test_assess_full_comparison_reuse_ready_with_completed_matlab_only(tmp_path: Path):
    run_root = tmp_path / "20260413_121500_comparison"
    metadata_dir = run_root / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(tmp_path / "input.tif")}}),
        encoding="utf-8",
    )
    (metadata_dir / "comparison_params.normalized.json").write_text(
        json.dumps(
            {
                "edge_method": "tracing",
                "comparison_exact_network": True,
                "python_parity_rerun_from": "edges",
            }
        ),
        encoding="utf-8",
    )
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260413-121500"
    batch_dir.mkdir(parents=True)
    (metadata_dir / "matlab_status.json").write_text(
        json.dumps(
            {
                "matlab_resume_mode": "complete-noop",
                "matlab_batch_complete": True,
                "matlab_batch_folder": str(batch_dir),
            }
        ),
        encoding="utf-8",
    )

    report = assess_loop_request(
        run_root,
        loop_kind="full_comparison",
        input_path=tmp_path / "input.tif",
        params={
            "edge_method": "tracing",
            "comparison_exact_network": True,
            "python_parity_rerun_from": "edges",
        },
    )

    assert report.verdict == "reuse_ready"
    assert report.safe_to_reuse is True
    assert report.requires_fresh_matlab is False


def test_assess_full_comparison_requires_fresh_matlab_without_completed_status(tmp_path: Path):
    run_root = tmp_path / "20260413_123000_comparison"
    metadata_dir = run_root / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(tmp_path / "input.tif")}}),
        encoding="utf-8",
    )
    (metadata_dir / "comparison_params.normalized.json").write_text(
        json.dumps(
            {
                "edge_method": "tracing",
                "comparison_exact_network": True,
                "python_parity_rerun_from": "edges",
            }
        ),
        encoding="utf-8",
    )

    report = assess_loop_request(
        run_root,
        loop_kind="full_comparison",
        input_path=tmp_path / "input.tif",
        params={
            "edge_method": "tracing",
            "comparison_exact_network": True,
            "python_parity_rerun_from": "edges",
        },
    )

    assert report.verdict == "fresh_matlab_required"
    assert report.requires_fresh_matlab is True
