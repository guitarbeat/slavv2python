from __future__ import annotations

from typing import Any

from .constants import PIPELINE_STAGES, PREPROCESS_STAGE, STAGE_WEIGHTS, STATUS_COMPLETED
from .models import RunSnapshot, StageSnapshot


def target_stage_progress(snapshot: RunSnapshot) -> float:
    """Return progress toward the user-selected pipeline target."""
    if snapshot.target_stage not in PIPELINE_STAGES:
        return float(snapshot.overall_progress)
    index = PIPELINE_STAGES.index(snapshot.target_stage)
    selected = PIPELINE_STAGES[: index + 1]
    total = STAGE_WEIGHTS[PREPROCESS_STAGE] + sum(STAGE_WEIGHTS[stage] for stage in selected)
    preprocess_stage = snapshot.stages.get(PREPROCESS_STAGE, StageSnapshot(name=PREPROCESS_STAGE))
    preprocess_done = bool(snapshot.artifacts.get("preprocess_done")) or (
        preprocess_stage.status == STATUS_COMPLETED
    )
    progress = STAGE_WEIGHTS[PREPROCESS_STAGE] if preprocess_done else 0.0
    for stage in selected:
        progress += STAGE_WEIGHTS[stage] * snapshot.stages.get(stage, StageSnapshot(stage)).progress
    return float(max(0.0, min(1.0, progress / total)))


def _stage_status_line(stage_name: str, stage: StageSnapshot) -> str:
    parts = [f"  - {stage_name}: {stage.status}", f"{stage.progress * 100:.1f}%"]
    if stage.resumed:
        parts.append("resumed")
    if stage.substage:
        parts.append(f"substage={stage.substage}")
    if stage.units_total:
        parts.append(f"units={stage.units_completed}/{stage.units_total}")
    if stage.detail:
        parts.append(stage.detail)
    return " | ".join(parts)


def _optional_task_status_line(name: str, task: Any) -> str:
    parts = [f"  - {name}: {task.status}", f"{task.progress * 100:.1f}%"]
    if task.detail:
        parts.append(task.detail)
    return " | ".join(parts)


def build_status_lines(snapshot: RunSnapshot) -> list[str]:
    """Create a human-readable status summary for CLI output."""
    lines = [
        f"Run ID: {snapshot.run_id}",
        f"Status: {snapshot.status}",
        f"Target stage: {snapshot.target_stage}",
        f"Current stage: {snapshot.current_stage or '(idle)'}",
        f"Overall progress: {snapshot.overall_progress * 100:.1f}%",
        f"Target progress: {target_stage_progress(snapshot) * 100:.1f}%",
        f"Elapsed: {snapshot.elapsed_seconds:.1f}s",
    ]
    if snapshot.eta_seconds is not None:
        lines.append(f"ETA: {snapshot.eta_seconds:.1f}s")
    lines.extend(("", "Stages:"))
    for stage_name in PIPELINE_STAGES:
        stage = snapshot.stages.get(stage_name, StageSnapshot(name=stage_name))
        lines.append(_stage_status_line(stage_name, stage))
    if snapshot.optional_tasks:
        lines.extend(("", "Optional tasks:"))
        lines.extend(
            _optional_task_status_line(name, task)
            for name, task in sorted(snapshot.optional_tasks.items())
        )
    if snapshot.errors:
        lines.extend(("", "Errors:"))
        lines.extend(
            f"  - {error.get('stage', 'run')}: {error.get('message', '')}"
            for error in snapshot.errors[-5:]
        )
    return lines
