"""Comprehensive tests for runtime helpers including layout, lifecycle, progress, reset, and status rendering."""

from __future__ import annotations

import calendar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from slavv_python.runtime import RunContext
from slavv_python.runtime.layout import resolve_run_layout
from slavv_python.runtime.lifecycle import (
    begin_stage_snapshot,
    complete_stage_snapshot,
    fail_stage_snapshot,
    finalize_run_snapshot,
    mark_preprocess_complete_snapshot,
    update_optional_task_snapshot,
    update_stage_snapshot,
)
from slavv_python.runtime.models import RunSnapshot, StageSnapshot
from slavv_python.runtime.progress import (
    calculate_overall_progress,
    estimate_run_eta,
    estimate_stage_eta,
    parse_run_time,
    preprocess_complete,
)
from slavv_python.runtime.reset import (
    clear_stage_runtime_artifacts,
    remove_stage_dir_contents,
    reset_stage_snapshots,
)
from slavv_python.runtime.status import build_status_lines
from tests.support.run_state_builders import build_run_context


class _DummyController:
    def __init__(self, stage_dir: Path) -> None:
        self.stage_dir = stage_dir
        self.checkpoint_path = stage_dir.parent / "checkpoint_energy.pkl"
        self.manifest_path = stage_dir / "stage_manifest.json"
        self.state_path = stage_dir / "resume_state.json"


# ==============================================================================
# Layout Helpers
# ==============================================================================


@pytest.mark.unit
def test_resolve_run_layout_structured_uses_staged_directories(tmp_path):
    layout = resolve_run_layout(run_dir=tmp_path / "run")
    assert layout.run_root == tmp_path / "run"
    assert layout.refs_dir == tmp_path / "run" / "00_Refs"
    assert layout.snapshot_path == tmp_path / "run" / "99_Metadata" / "run_snapshot.json"


@pytest.mark.unit
def test_resolve_run_layout_requires_run_dir():
    with pytest.raises(ValueError, match="run_dir is required"):
        resolve_run_layout(run_dir=None)


# ==============================================================================
# Lifecycle Helpers
# ==============================================================================


@pytest.mark.unit
def test_mark_preprocess_complete_snapshot_marks_artifacts_and_progress():
    snapshot = RunSnapshot(run_id="run-123")
    mark_preprocess_complete_snapshot(snapshot, overall_progress=0.2)
    preprocess = snapshot.stages["preprocess"]
    assert preprocess.status == "completed"
    assert snapshot.artifacts["preprocess_done"] == "true"


@pytest.mark.unit
def test_begin_stage_snapshot_sets_running_state():
    snapshot = RunSnapshot(run_id="run-123")
    stage = begin_stage_snapshot(
        snapshot,
        stage="energy",
        detail="Starting energy",
        units_total=4,
        units_completed=1,
    )
    assert stage.status == "running"
    assert snapshot.current_stage == "energy"


@pytest.mark.unit
def test_update_stage_snapshot_updates_eta_and_last_event():
    snapshot = RunSnapshot(
        run_id="run-123",
        stages={"energy": StageSnapshot(name="energy", units_total=4, units_completed=1)},
    )
    stage = update_stage_snapshot(snapshot, stage="energy", units_completed=3, detail="Almost done")
    assert stage.progress == 0.75
    assert snapshot.last_event == "Almost done"


@pytest.mark.unit
def test_complete_stage_snapshot_promotes_artifacts_and_completion():
    snapshot = RunSnapshot(
        run_id="run-123",
        stages={"energy": StageSnapshot(name="energy", units_total=4, units_completed=4)},
    )
    complete_stage_snapshot(
        snapshot,
        stage="energy",
        artifacts={"checkpoint": "checkpoint_energy.pkl"},
    )
    assert snapshot.artifacts["energy.checkpoint"] == "checkpoint_energy.pkl"


@pytest.mark.unit
def test_fail_stage_snapshot_marks_run_failed():
    snapshot = RunSnapshot(run_id="run-123")
    fail_stage_snapshot(snapshot, stage="edges", message="trace failed")
    assert snapshot.status == "failed"
    assert snapshot.errors[-1]["message"] == "trace failed"


@pytest.mark.unit
def test_update_optional_task_snapshot_marks_completion():
    snapshot = RunSnapshot(run_id="run-123")
    task = update_optional_task_snapshot(
        snapshot,
        name="exports",
        status="completed",
        artifacts={"json": "network.json"},
    )
    assert task.progress == 1.0
    assert task.artifacts["json"] == "network.json"


@pytest.mark.unit
def test_finalize_run_snapshot_marks_target_completion():
    snapshot = RunSnapshot(run_id="run-123", overall_progress=0.8)
    finalize_run_snapshot(snapshot, stop_after="edges")
    assert snapshot.status == "completed_target"
    assert snapshot.current_stage == "edges"


# ==============================================================================
# Progress Helpers
# ==============================================================================


@pytest.mark.unit
def test_parse_run_time_uses_utc_epoch():
    timestamp = "2026-03-27T12:00:00Z"
    parsed = parse_run_time(timestamp)
    assert parsed == calendar.timegm((2026, 3, 27, 12, 0, 0))


@pytest.mark.unit
def test_estimate_stage_eta_updates_elapsed_seconds():
    stage = StageSnapshot(
        name="energy",
        progress=0.5,
        started_at="2026-03-27T12:00:00Z",
    )
    eta = estimate_stage_eta(stage, now=calendar.timegm((2026, 3, 27, 12, 0, 10)))
    assert eta == 10.0
    assert stage.elapsed_seconds == 10.0


@pytest.mark.unit
def test_estimate_run_eta_updates_snapshot_elapsed_seconds():
    snapshot = RunSnapshot(
        run_id="run-123",
        created_at="2026-03-27T12:00:00Z",
        overall_progress=0.25,
    )
    eta = estimate_run_eta(snapshot, now=calendar.timegm((2026, 3, 27, 12, 0, 20)))
    assert eta == 60.0
    assert snapshot.elapsed_seconds == 20.0


@pytest.mark.unit
def test_preprocess_complete_checks_snapshot_artifacts_and_stage_status():
    stages = {"preprocess": StageSnapshot(name="preprocess")}
    snapshot = RunSnapshot(run_id="run-123", artifacts={"preprocess_done": "true"})
    assert preprocess_complete(stages, snapshot=snapshot) is True


@pytest.mark.unit
def test_calculate_overall_progress_uses_stage_weights_and_preprocess_flag():
    stages = {
        "energy": StageSnapshot(name="energy", progress=1.0),
        "vertices": StageSnapshot(name="vertices", progress=0.5),
        "edges": StageSnapshot(name="edges", progress=0.0),
        "network": StageSnapshot(name="network", progress=0.0),
    }
    progress = calculate_overall_progress(stages, preprocess_done=True)
    assert 0.0 < progress < 1.0


# ==============================================================================
# Reset Helpers
# ==============================================================================


@pytest.mark.unit
def test_remove_stage_dir_contents_removes_files_and_nested_directories(tmp_path):
    stage_dir = tmp_path / "stage"
    nested_dir = stage_dir / "nested" / "deeper"
    nested_dir.mkdir(parents=True)
    (stage_dir / "artifact.txt").write_text("artifact", encoding="utf-8")
    remove_stage_dir_contents(stage_dir)
    assert stage_dir.exists()
    assert list(stage_dir.iterdir()) == []


@pytest.mark.unit
def test_clear_stage_runtime_artifacts_removes_checkpoint_manifest_and_state(tmp_path):
    stage_dir = tmp_path / "stages" / "energy"
    stage_dir.mkdir(parents=True)
    controller = _DummyController(stage_dir)
    controller.checkpoint_path.write_text("checkpoint", encoding="utf-8")
    controller.manifest_path.write_text("manifest", encoding="utf-8")
    controller.state_path.write_text("state", encoding="utf-8")
    clear_stage_runtime_artifacts(controller)
    assert not controller.checkpoint_path.exists()
    assert list(stage_dir.iterdir()) == []


@pytest.mark.unit
def test_reset_stage_snapshots_reinitializes_requested_and_later_stages():
    stages = {
        "energy": StageSnapshot(name="energy", status="completed", progress=1.0),
        "vertices": StageSnapshot(name="vertices", status="running", progress=0.5),
        "edges": StageSnapshot(name="edges", status="completed", progress=1.0),
        "network": StageSnapshot(name="network", status="failed", progress=0.3),
    }
    affected = reset_stage_snapshots(stages, start_stage="vertices")
    assert affected == ["vertices", "edges", "network"]
    assert stages["vertices"].status == "pending"


# ==============================================================================
# Status Rendering Tests
# ==============================================================================


@pytest.mark.unit
def test_parse_time_uses_utc_epoch_context():
    timestamp = "2026-03-27T12:00:00Z"
    parsed = RunContext._parse_time(timestamp)
    assert parsed == calendar.timegm((2026, 3, 27, 12, 0, 0))


@pytest.mark.unit
def test_build_status_lines_include_optional_tasks(tmp_path):
    run_dir = tmp_path / "run"
    context = build_run_context(
        run_dir,
        target_stage="network",
    )
    context.update_optional_task(
        "export_network",
        status="running",
        detail="Writing JSON and CSV exports.",
    )
    lines = build_status_lines(context.snapshot)
    rendered = "\n".join(lines)
    assert "Optional tasks:" in rendered
    assert "export_network: running" in rendered
