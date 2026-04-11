"""Tests for file-backed resumable run state."""

from __future__ import annotations

import calendar
import json

import joblib
import numpy as np
import pytest

from slavv.core import SLAVVProcessor
from slavv.runtime import RunContext, load_run_snapshot
from slavv.runtime.run_state import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_FAILED,
    STATUS_PENDING,
    atomic_write_json,
    build_status_lines,
    load_legacy_run_snapshot,
    target_stage_progress,
)


def test_atomic_write_json_replaces_previous_content(tmp_path):
    path = tmp_path / "snapshot.json"

    atomic_write_json(path, {"stage": "energy", "progress": 0.25})
    atomic_write_json(path, {"stage": "network", "progress": 1.0})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "progress": 1.0,
        "stage": "network",
    }


def test_run_context_persists_snapshot_lifecycle(tmp_path):
    run_dir = tmp_path / "run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="edges",
        provenance={"source": "unit-test"},
    )

    context.mark_preprocess_complete()
    energy_stage = context.stage("energy")
    energy_stage.begin(detail="Computing energy", units_total=4, units_completed=1)
    energy_stage.update(units_completed=2, detail="Halfway through energy")
    energy_stage.save_checkpoint({"energy": "ready"})
    energy_stage.complete(detail="Energy complete", artifacts={"energy_field": "energy.npy"})

    context.update_optional_task("exports", status="running", detail="Preparing exports")
    context.update_optional_task(
        "exports",
        status="completed",
        detail="Exported requested outputs",
        artifacts={"json": "network.json"},
    )
    context.finalize_run(stop_after="edges")

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_COMPLETED_TARGET
    assert snapshot.target_stage == "edges"
    assert snapshot.stages["preprocess"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].artifacts["checkpoint"].endswith("checkpoint_energy.pkl")
    assert snapshot.stages["energy"].artifacts["energy_field"] == "energy.npy"
    assert snapshot.optional_tasks["exports"].status == STATUS_COMPLETED
    assert snapshot.optional_tasks["exports"].artifacts["json"] == "network.json"
    assert target_stage_progress(snapshot) >= snapshot.overall_progress


def test_resume_guard_blocks_mismatched_input(tmp_path):
    run_dir = tmp_path / "blocked_run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )

    with pytest.raises(RuntimeError, match="Resume blocked"):
        context.ensure_resume_allowed(
            input_fingerprint="input-b",
            params_fingerprint="params-a",
        )

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_BLOCKED
    assert "fingerprint changed" in snapshot.last_event
    assert snapshot.errors


def test_force_rerun_from_energy_resets_pipeline_state(tmp_path):
    run_dir = tmp_path / "reset_run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )
    energy_stage = context.stage("energy")
    energy_stage.save_checkpoint({"energy": "ready"})
    energy_stage.complete(detail="Energy complete")

    assert energy_stage.checkpoint_path.exists()

    context.ensure_resume_allowed(
        input_fingerprint="input-b",
        params_fingerprint="params-a",
        force_rerun_from="energy",
    )

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_PENDING
    assert snapshot.input_fingerprint == "input-b"
    assert snapshot.stages["energy"].status == STATUS_PENDING
    assert not energy_stage.checkpoint_path.exists()


def test_process_image_blocks_reuse_when_parameters_change(tmp_path):
    processor = SLAVVProcessor()
    image = np.zeros((4, 4, 4), dtype=np.float32)
    run_dir = tmp_path / "run"

    processor.process_image(
        image,
        {"radius_of_smallest_vessel_in_microns": 1.5},
        run_dir=str(run_dir),
        stop_after="energy",
    )
    first_snapshot = load_run_snapshot(run_dir)
    assert first_snapshot is not None
    original_params_fingerprint = first_snapshot.params_fingerprint

    with pytest.raises(RuntimeError, match="Resume blocked"):
        processor.process_image(
            image,
            {"radius_of_smallest_vessel_in_microns": 2.0},
            run_dir=str(run_dir),
            stop_after="energy",
        )

    blocked_snapshot = load_run_snapshot(run_dir)
    assert blocked_snapshot is not None
    assert blocked_snapshot.status == STATUS_BLOCKED
    assert blocked_snapshot.params_fingerprint == original_params_fingerprint


def test_stage_controller_rejects_unknown_stage(tmp_path):
    context = RunContext(
        run_dir=tmp_path / "run",
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )

    with pytest.raises(ValueError, match="stage must be one of"):
        context.stage("preprocess")


def test_legacy_checkpoints_bootstrap_snapshot(tmp_path):
    checkpoint_dir = tmp_path / "legacy_checkpoints"
    checkpoint_dir.mkdir()
    joblib.dump({"energy": True}, checkpoint_dir / "checkpoint_energy.pkl")
    joblib.dump({"vertices": True}, checkpoint_dir / "checkpoint_vertices.pkl")

    context = RunContext(
        checkpoint_dir=checkpoint_dir,
        target_stage="network",
        legacy=True,
    )

    snapshot = context.snapshot
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["vertices"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].resumed is True
    assert snapshot.stages["edges"].status == STATUS_PENDING
    assert load_run_snapshot(checkpoint_dir) is not None


def test_from_existing_preserves_existing_target_stage(tmp_path):
    run_dir = tmp_path / "run"
    RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="edges",
    )

    reopened = RunContext.from_existing(run_dir)

    assert reopened.snapshot.target_stage == "edges"
    assert load_run_snapshot(run_dir).target_stage == "edges"


def test_load_legacy_run_snapshot_is_read_only(tmp_path):
    checkpoint_dir = tmp_path / "legacy_checkpoints"
    checkpoint_dir.mkdir()
    joblib.dump({"energy": True}, checkpoint_dir / "checkpoint_energy.pkl")

    snapshot = load_legacy_run_snapshot(checkpoint_dir)
    repeated = load_legacy_run_snapshot(checkpoint_dir)

    assert snapshot is not None
    assert repeated is not None
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["preprocess"].status == STATUS_COMPLETED
    assert snapshot.overall_progress == pytest.approx(0.40)
    assert snapshot.run_id == repeated.run_id
    assert not (checkpoint_dir / "run_snapshot.json").exists()
    assert not (checkpoint_dir / "stage_state").exists()


def test_process_image_rejects_invalid_stop_after():
    processor = SLAVVProcessor()
    image = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="stop_after must be one of"):
        processor.process_image(image, {}, stop_after="bogus")


def test_preprocess_failure_marks_run_failed(tmp_path, monkeypatch):
    processor = SLAVVProcessor()
    image = np.zeros((4, 4, 4), dtype=np.float32)
    run_dir = tmp_path / "run"

    def _boom(_image, _parameters):
        raise RuntimeError("preprocess exploded")

    monkeypatch.setattr("slavv.core.pipeline.utils.preprocess_image", _boom)

    with pytest.raises(RuntimeError, match="preprocess exploded"):
        processor.process_image(image, {}, run_dir=str(run_dir))

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_FAILED
    assert snapshot.current_stage == "preprocess"
    assert snapshot.stages["preprocess"].status == STATUS_FAILED


def test_parse_time_uses_utc_epoch():
    timestamp = "2026-03-27T12:00:00Z"

    parsed = RunContext._parse_time(timestamp)

    assert parsed == calendar.timegm((2026, 3, 27, 12, 0, 0))


def test_build_status_lines_includes_matlab_resume_details(tmp_path):
    run_dir = tmp_path / "run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )
    context.update_optional_task(
        "matlab_status",
        status="failed",
        detail="Rerun will reuse batch_260401-140000 but restart energy from the stage boundary.",
        artifacts={
            "batch_folder": "C:\\temp\\batch_260401-140000",
            "resume_mode": "restart-current-stage",
            "last_completed_stage": "(none)",
            "next_stage": "energy",
            "rerun_prediction": "Rerun will reuse batch_260401-140000 but restart energy from the stage boundary.",
            "python_force_rerun_from": "edges",
            "failure_summary_file": "C:\\temp\\matlab_failure_summary.json",
        },
    )

    lines = build_status_lines(context.snapshot)
    rendered = "\n".join(lines)

    assert "MATLAB resume:" in rendered
    assert "Batch folder: C:\\temp\\batch_260401-140000" in rendered
    assert "Resume mode: restart-current-stage | last completed=(none) | next=energy" in rendered
    assert (
        "Prediction: Rerun will reuse batch_260401-140000 but restart energy from the stage boundary."
        in rendered
    )
    assert "Python rerun from: edges" in rendered
