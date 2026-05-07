"""Tests for run-state snapshot lifecycle helpers."""

from __future__ import annotations

from source.runtime.run_tracking.lifecycle import (
    begin_stage_snapshot,
    complete_stage_snapshot,
    fail_stage_snapshot,
    finalize_run_snapshot,
    mark_preprocess_complete_snapshot,
    update_optional_task_snapshot,
    update_stage_snapshot,
)
from source.runtime.run_tracking.models import RunSnapshot, StageSnapshot


def test_mark_preprocess_complete_snapshot_marks_artifacts_and_progress():
    snapshot = RunSnapshot(run_id="run-123")

    mark_preprocess_complete_snapshot(snapshot, overall_progress=0.2)

    preprocess = snapshot.stages["preprocess"]
    assert preprocess.status == "completed"
    assert preprocess.progress == 1.0
    assert snapshot.artifacts["preprocess_done"] == "true"
    assert snapshot.overall_progress == 0.2


def test_begin_stage_snapshot_sets_running_state():
    snapshot = RunSnapshot(run_id="run-123")

    stage = begin_stage_snapshot(
        snapshot,
        stage="energy",
        detail="Starting energy",
        units_total=4,
        units_completed=1,
        substage="chunk-1",
        resumed=True,
    )

    assert stage.status == "running"
    assert stage.progress == 0.25
    assert stage.resumed is True
    assert snapshot.current_stage == "energy"
    assert snapshot.current_detail == "Starting energy"
    assert snapshot.status == "running"


def test_update_stage_snapshot_updates_eta_and_last_event():
    snapshot = RunSnapshot(
        run_id="run-123",
        stages={"energy": StageSnapshot(name="energy", units_total=4, units_completed=1)},
    )

    stage = update_stage_snapshot(
        snapshot,
        stage="energy",
        units_completed=3,
        detail="Almost done",
    )

    assert stage.progress == 0.75
    assert snapshot.last_event == "Almost done"
    assert snapshot.current_detail == "Almost done"


def test_complete_stage_snapshot_promotes_artifacts_and_completion():
    snapshot = RunSnapshot(
        run_id="run-123",
        stages={"energy": StageSnapshot(name="energy", units_total=4, units_completed=4)},
    )

    stage = complete_stage_snapshot(
        snapshot,
        stage="energy",
        detail="Energy ready",
        artifacts={"checkpoint": "checkpoint_energy.pkl"},
    )

    assert stage.status == "completed"
    assert stage.progress == 1.0
    assert snapshot.artifacts["energy.checkpoint"] == "checkpoint_energy.pkl"
    assert snapshot.last_event == "Energy ready"


def test_fail_stage_snapshot_marks_run_failed():
    snapshot = RunSnapshot(run_id="run-123")

    stage = fail_stage_snapshot(snapshot, stage="edges", message="trace failed")

    assert stage.status == "failed"
    assert snapshot.status == "failed"
    assert snapshot.current_stage == "edges"
    assert snapshot.errors[-1]["message"] == "trace failed"


def test_update_optional_task_snapshot_marks_completion():
    snapshot = RunSnapshot(run_id="run-123")

    task = update_optional_task_snapshot(
        snapshot,
        name="exports",
        status="completed",
        detail="Ready",
        artifacts={"json": "network.json"},
    )

    assert task.progress == 1.0
    assert task.artifacts["json"] == "network.json"
    assert snapshot.last_event == "Ready"


def test_finalize_run_snapshot_marks_target_completion():
    snapshot = RunSnapshot(run_id="run-123", overall_progress=0.8)

    finalize_run_snapshot(snapshot, stop_after="edges")

    assert snapshot.status == "completed_target"
    assert snapshot.current_stage == "edges"
    assert snapshot.eta_seconds == 0.0
