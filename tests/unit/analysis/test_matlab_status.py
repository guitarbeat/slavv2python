"""Tests for normalized MATLAB resume-status parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slavv.evaluation.matlab_status import inspect_matlab_status

if TYPE_CHECKING:
    from pathlib import Path


def test_inspect_matlab_status_reports_fresh_run_when_no_batch_exists(tmp_path: Path):
    output_dir = tmp_path / "matlab_results"
    output_dir.mkdir()

    report = inspect_matlab_status(output_dir, input_file=tmp_path / "input.tif")

    assert report.matlab_resume_mode == "fresh"
    assert report.matlab_batch_folder == ""
    assert report.matlab_next_stage == "energy"
    assert "create a new batch" in report.matlab_rerun_prediction


def test_inspect_matlab_status_reports_stage_boundary_resume(
    tmp_path: Path,
    matlab_artifact_builder,
):
    input_file = tmp_path / "input.tif"
    artifacts = matlab_artifact_builder(
        tmp_path / "matlab_results",
        input_file=input_file,
        batch_timestamp="260401-120000",
        completed_stages=("energy", "vertices"),
    )

    report = inspect_matlab_status(artifacts["output_dir"], input_file=input_file)

    assert report.matlab_resume_mode == "resume-stage"
    assert report.matlab_last_completed_stage == "vertices"
    assert report.matlab_next_stage == "edges"
    assert report.matlab_partial_stage_artifacts_present is False
    assert "start at edges" in report.matlab_rerun_prediction


def test_inspect_matlab_status_reports_complete_noop_for_finished_batch(
    tmp_path: Path,
    matlab_artifact_builder,
):
    input_file = tmp_path / "input.tif"
    artifacts = matlab_artifact_builder(
        tmp_path / "matlab_results",
        input_file=input_file,
        batch_timestamp="260401-130000",
        completed_stages=("energy", "vertices", "edges", "network"),
    )

    report = inspect_matlab_status(artifacts["output_dir"], input_file=input_file)

    assert report.matlab_resume_mode == "complete-noop"
    assert report.matlab_last_completed_stage == "network"
    assert report.matlab_next_stage == ""
    assert "already complete" in report.matlab_rerun_prediction


def test_inspect_matlab_status_detects_mid_stage_restart_and_failure_summary(
    tmp_path: Path,
    matlab_artifact_builder,
):
    input_file = tmp_path / "input.tif"
    artifacts = matlab_artifact_builder(
        tmp_path / "matlab_results",
        input_file=input_file,
        batch_timestamp="260401-140000",
        completed_stages=("energy",),
        running_status="running:energy",
        partial_stage="energy",
        chunk_names=("1 of 121", "2 of 121", "10 of 121"),
        log_lines=[
            "Running Energy workflow...",
            "Starting parallel pool (parpool) using the 'local' profile ...",
            "Connected to the parallel pool (number of workers: 4).",
            "ERROR: MATLAB error Exit Status: 0x00000001",
        ],
    )

    report = inspect_matlab_status(artifacts["output_dir"], input_file=input_file)

    assert report.matlab_resume_mode == "restart-current-stage"
    assert report.matlab_batch_folder.endswith("batch_260401-140000")
    assert report.matlab_next_stage == "energy"
    assert report.matlab_partial_stage_artifacts_present is True
    assert report.matlab_partial_stage_name == "energy"
    assert report.matlab_partial_stage_details["observed_chunk_count"] == 3
    assert report.matlab_partial_stage_details["expected_chunk_count"] == 121
    assert report.failure_summary == "ERROR: MATLAB error Exit Status: 0x00000001"
    assert report.stale_running_snapshot_suspected is True
    assert "restart energy" in report.matlab_rerun_prediction
