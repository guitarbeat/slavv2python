"""Pure progress and ETA helpers for run-state bookkeeping."""

from __future__ import annotations

import calendar
import time
from typing import TYPE_CHECKING

from .constants import PIPELINE_STAGES, PREPROCESS_STAGE, STAGE_WEIGHTS, STATUS_COMPLETED
from .models import StageSnapshot

if TYPE_CHECKING:
    from .models import RunSnapshot


def parse_run_time(timestamp: str | None) -> float | None:
    """Parse an ISO-8601 UTC timestamp into epoch seconds."""
    if not timestamp:
        return None
    try:
        return calendar.timegm(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def estimate_stage_eta(stage_snapshot: StageSnapshot, *, now: float | None = None) -> float | None:
    """Estimate stage ETA from current progress and start time."""
    if stage_snapshot.progress <= 0.0 or stage_snapshot.started_at is None:
        return None
    started = parse_run_time(stage_snapshot.started_at)
    if started is None:
        return None
    current_time = time.time() if now is None else now
    elapsed = max(0.0, current_time - started)
    stage_snapshot.elapsed_seconds = elapsed
    return float(elapsed * (1.0 - stage_snapshot.progress) / stage_snapshot.progress)


def estimate_run_eta(snapshot: RunSnapshot, *, now: float | None = None) -> float | None:
    """Estimate whole-run ETA from the current overall progress."""
    if snapshot.overall_progress <= 0.0:
        return None
    created = parse_run_time(snapshot.created_at)
    if created is None:
        return None
    current_time = time.time() if now is None else now
    elapsed = max(0.0, current_time - created)
    snapshot.elapsed_seconds = elapsed
    return float(elapsed * (1.0 - snapshot.overall_progress) / snapshot.overall_progress)


def calculate_overall_progress(
        stages: dict[str, StageSnapshot],
        *,
        preprocess_done: bool,
) -> float:
    """Return normalized overall pipeline progress across preprocess and stage weights."""
    total = STAGE_WEIGHTS[PREPROCESS_STAGE] + sum(STAGE_WEIGHTS[stage] for stage in PIPELINE_STAGES)
    progress = STAGE_WEIGHTS[PREPROCESS_STAGE] if preprocess_done else 0.0
    for stage_name in PIPELINE_STAGES:
        progress += (
                STAGE_WEIGHTS[stage_name] * stages.get(stage_name, StageSnapshot(stage_name)).progress
        )
    return float(max(0.0, min(1.0, progress / total)))


def preprocess_complete(stages: dict[str, StageSnapshot], *, snapshot: RunSnapshot | None) -> bool:
    """Return whether preprocess should count as complete for overall progress."""
    preprocess_stage = stages.get(PREPROCESS_STAGE, StageSnapshot(name=PREPROCESS_STAGE))
    return bool(snapshot and snapshot.artifacts.get("preprocess_done")) or (
            preprocess_stage.status == STATUS_COMPLETED
    )


__all__ = [
    "calculate_overall_progress",
    "estimate_run_eta",
    "estimate_stage_eta",
    "parse_run_time",
    "preprocess_complete",
]
