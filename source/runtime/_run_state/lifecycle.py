"""Snapshot lifecycle mutation helpers for run-state bookkeeping."""

from __future__ import annotations

from .constants import (
    PREPROCESS_STAGE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from .models import RunSnapshot, StageSnapshot, TaskSnapshot, _now_iso


def mark_preprocess_complete_snapshot(snapshot: RunSnapshot, *, overall_progress: float) -> None:
    """Mark preprocess complete on the snapshot."""
    stage_snapshot = snapshot.stages.setdefault(
        PREPROCESS_STAGE, StageSnapshot(name=PREPROCESS_STAGE)
    )
    if stage_snapshot.started_at is None:
        stage_snapshot.started_at = _now_iso()
    stage_snapshot.status = STATUS_COMPLETED
    stage_snapshot.progress = 1.0
    stage_snapshot.units_total = max(stage_snapshot.units_total, 1)
    stage_snapshot.units_completed = stage_snapshot.units_total
    stage_snapshot.detail = "Preprocessing complete"
    stage_snapshot.updated_at = _now_iso()
    stage_snapshot.completed_at = _now_iso()
    snapshot.artifacts["preprocess_done"] = "true"
    snapshot.overall_progress = overall_progress
    snapshot.last_event = "Preprocessing complete"


def begin_stage_snapshot(
    snapshot: RunSnapshot,
    *,
    stage: str,
    detail: str = "",
    units_total: int = 0,
    units_completed: int = 0,
    substage: str = "",
    resumed: bool = False,
) -> StageSnapshot:
    """Begin a stage and return the mutated stage snapshot."""
    stage_snapshot = snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
    if stage_snapshot.started_at is None:
        stage_snapshot.started_at = _now_iso()
    stage_snapshot.status = STATUS_RUNNING
    stage_snapshot.updated_at = _now_iso()
    stage_snapshot.detail = detail
    stage_snapshot.substage = substage
    stage_snapshot.units_total = units_total or stage_snapshot.units_total
    stage_snapshot.units_completed = units_completed
    stage_snapshot.resumed = resumed or stage_snapshot.resumed
    if stage_snapshot.units_total > 0:
        stage_snapshot.progress = min(
            1.0, stage_snapshot.units_completed / stage_snapshot.units_total
        )
    snapshot.current_stage = stage
    snapshot.status = STATUS_RUNNING
    snapshot.last_event = detail or f"Running {stage}"
    return stage_snapshot


def update_stage_snapshot(
    snapshot: RunSnapshot,
    *,
    stage: str,
    detail: str | None = None,
    units_total: int | None = None,
    units_completed: int | None = None,
    progress: float | None = None,
    substage: str | None = None,
    resumed: bool | None = None,
) -> StageSnapshot:
    """Update an in-flight stage and return the mutated stage snapshot."""
    stage_snapshot = snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
    stage_snapshot.status = STATUS_RUNNING
    stage_snapshot.updated_at = _now_iso()
    if detail is not None:
        stage_snapshot.detail = detail
    if substage is not None:
        stage_snapshot.substage = substage
    if units_total is not None:
        stage_snapshot.units_total = units_total
    if units_completed is not None:
        stage_snapshot.units_completed = units_completed
    if resumed is not None:
        stage_snapshot.resumed = resumed
    if progress is not None:
        stage_snapshot.progress = max(0.0, min(1.0, progress))
    elif stage_snapshot.units_total > 0:
        stage_snapshot.progress = min(
            1.0, stage_snapshot.units_completed / stage_snapshot.units_total
        )
    snapshot.current_stage = stage
    snapshot.status = STATUS_RUNNING
    snapshot.last_event = stage_snapshot.detail or f"Running {stage}"
    return stage_snapshot


def complete_stage_snapshot(
    snapshot: RunSnapshot,
    *,
    stage: str,
    detail: str = "",
    artifacts: dict[str, str] | None = None,
    resumed: bool | None = None,
) -> StageSnapshot:
    """Complete a stage and return the mutated stage snapshot."""
    stage_snapshot = snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
    stage_snapshot.status = STATUS_COMPLETED
    stage_snapshot.progress = 1.0
    stage_snapshot.updated_at = _now_iso()
    stage_snapshot.completed_at = _now_iso()
    stage_snapshot.units_total = max(stage_snapshot.units_total, stage_snapshot.units_completed, 1)
    stage_snapshot.units_completed = stage_snapshot.units_total
    if detail:
        stage_snapshot.detail = detail
    if artifacts:
        stage_snapshot.artifacts.update(artifacts)
        snapshot.artifacts.update({f"{stage}.{k}": v for k, v in artifacts.items()})
    if resumed is not None:
        stage_snapshot.resumed = resumed
    snapshot.last_event = detail or f"Completed {stage}"
    return stage_snapshot


def fail_stage_snapshot(snapshot: RunSnapshot, *, stage: str, message: str) -> StageSnapshot:
    """Mark a stage and the run as failed."""
    stage_snapshot = snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
    stage_snapshot.status = STATUS_FAILED
    stage_snapshot.updated_at = _now_iso()
    stage_snapshot.detail = message
    snapshot.status = STATUS_FAILED
    snapshot.current_stage = stage
    snapshot.last_event = message
    snapshot.errors.append({"stage": stage, "message": message, "at": _now_iso()})
    return stage_snapshot


def update_optional_task_snapshot(
    snapshot: RunSnapshot,
    *,
    name: str,
    status: str,
    detail: str = "",
    progress: float | None = None,
    artifacts: dict[str, str] | None = None,
) -> TaskSnapshot:
    """Update an optional task and return the mutated task snapshot."""
    task = snapshot.optional_tasks.setdefault(name, TaskSnapshot(name=name))
    if task.started_at is None and status == STATUS_RUNNING:
        task.started_at = _now_iso()
    task.status = status
    task.updated_at = _now_iso()
    task.detail = detail
    if progress is not None:
        task.progress = progress
    if status == STATUS_COMPLETED:
        task.progress = 1.0
        task.completed_at = _now_iso()
    if artifacts:
        task.artifacts.update(artifacts)
    snapshot.last_event = detail or f"Task {name}: {status}"
    return task


def finalize_run_snapshot(snapshot: RunSnapshot, *, stop_after: str | None = None) -> None:
    """Finalize the run snapshot for a full or target-stage completion."""
    if stop_after and stop_after != "network":
        snapshot.status = STATUS_COMPLETED_TARGET
    else:
        snapshot.status = STATUS_COMPLETED
        snapshot.overall_progress = 1.0
    snapshot.current_stage = stop_after or "network"
    snapshot.eta_seconds = 0.0
    snapshot.last_event = "Run completed"


__all__ = [
    "begin_stage_snapshot",
    "complete_stage_snapshot",
    "fail_stage_snapshot",
    "finalize_run_snapshot",
    "mark_preprocess_complete_snapshot",
    "update_optional_task_snapshot",
    "update_stage_snapshot",
]
