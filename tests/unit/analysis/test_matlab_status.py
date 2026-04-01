"""Tests for normalized MATLAB resume-status parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.io import savemat

from slavv.evaluation.matlab_status import inspect_matlab_status

if TYPE_CHECKING:
    from pathlib import Path


def _write_batch_settings(batch_folder: Path, input_file: Path, roi_name: str = "_r"):
    settings_dir = batch_folder / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "batch.mat",
        {
            "optional_input": np.array([str(input_file)], dtype=object),
            "ROI_names": np.array([roi_name], dtype=object),
        },
    )


def test_inspect_matlab_status_reports_fresh_run_when_no_batch_exists(tmp_path: Path):
    output_dir = tmp_path / "matlab_results"
    output_dir.mkdir()

    report = inspect_matlab_status(output_dir, input_file=tmp_path / "input.tif")

    assert report.matlab_resume_mode == "fresh"
    assert report.matlab_batch_folder == ""
    assert report.matlab_next_stage == "energy"
    assert "create a new batch" in report.matlab_rerun_prediction


def test_inspect_matlab_status_reports_stage_boundary_resume(tmp_path: Path):
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
    batch_folder = tmp_path / "matlab_results" / "batch_260401-120000"
    _write_batch_settings(batch_folder, input_file)
    data_dir = batch_folder / "data"
    vectors_dir = batch_folder / "vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "energy_260401-120000__r").write_text("", encoding="utf-8")
    (vectors_dir / "vertices_260401-120000__r.mat").write_text(
        "", encoding="utf-8"
    )
    (vectors_dir / "curated_vertices_260401-120000__r.mat").write_text(
        "", encoding="utf-8"
    )

    report = inspect_matlab_status(batch_folder.parent, input_file=input_file)

    assert report.matlab_resume_mode == "resume-stage"
    assert report.matlab_last_completed_stage == "vertices"
    assert report.matlab_next_stage == "edges"
    assert report.matlab_partial_stage_artifacts_present is False
    assert "start at edges" in report.matlab_rerun_prediction


def test_inspect_matlab_status_reports_complete_noop_for_finished_batch(tmp_path: Path):
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
    batch_folder = tmp_path / "matlab_results" / "batch_260401-130000"
    _write_batch_settings(batch_folder, input_file)
    data_dir = batch_folder / "data"
    vectors_dir = batch_folder / "vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "energy_260401-130000__r").write_text("", encoding="utf-8")
    (vectors_dir / "vertices_260401-130000__r.mat").write_text(
        "", encoding="utf-8"
    )
    (vectors_dir / "curated_vertices_260401-130000__r.mat").write_text(
        "", encoding="utf-8"
    )
    (vectors_dir / "edges_260401-130000__r.mat").write_text(
        "", encoding="utf-8"
    )
    (vectors_dir / "curated_edges_260401-130000__r.mat").write_text(
        "", encoding="utf-8"
    )
    (vectors_dir / "network_260401-130000__r.mat").write_text(
        "", encoding="utf-8"
    )

    report = inspect_matlab_status(batch_folder.parent, input_file=input_file)

    assert report.matlab_resume_mode == "complete-noop"
    assert report.matlab_last_completed_stage == "network"
    assert report.matlab_next_stage == ""
    assert "already complete" in report.matlab_rerun_prediction


def test_inspect_matlab_status_detects_mid_stage_restart_and_failure_summary(tmp_path: Path):
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
    output_dir = tmp_path / "matlab_results"
    batch_folder = output_dir / "batch_260401-140000"
    _write_batch_settings(batch_folder, input_file)
    data_dir = batch_folder / "data"
    vectors_dir = batch_folder / "vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "energy_260401-140000__r").write_text("", encoding="utf-8")
    chunk_dir = data_dir / "energy_260401-140000__r_chunks_octave_2_of_6"
    chunk_dir.mkdir()
    for name in ("1 of 121", "2 of 121", "10 of 121"):
        (chunk_dir / name).write_text("", encoding="utf-8")
    (output_dir / "matlab_resume_state.json").write_text(
        '{{"input_file":"{}","output_directory":"{}","batch_timestamp":"","batch_folder":"",'
        '"last_completed_stage":"","status":"running:energy","updated_at":"2026-04-01 12:27:34"}}'.format(
            str(input_file).replace("\\", "/"),
            str(output_dir).replace("\\", "/"),
        ),
        encoding="utf-8",
    )
    (output_dir / "matlab_run.log").write_text(
        "\n".join(
            [
                "Running Energy workflow...",
                "Starting parallel pool (parpool) using the 'local' profile ...",
                "Connected to the parallel pool (number of workers: 4).",
                "ERROR: MATLAB error Exit Status: 0x00000001",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_matlab_status(output_dir, input_file=input_file)

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
