"""Reusable builders for run-state and checkpoint test fixtures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import joblib
from source.runtime import RunContext

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def build_stage_snapshot_dict(
    name: str,
    *,
    status: str = "pending",
    progress: float = 0.0,
    resumed: bool = False,
    detail: str = "",
    substage: str = "",
    units_total: int = 0,
    units_completed: int = 0,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a stage snapshot payload compatible with ``RunSnapshot.from_dict``."""
    return {
        "name": name,
        "status": status,
        "progress": progress,
        "resumed": resumed,
        "detail": detail,
        "substage": substage,
        "units_total": units_total,
        "units_completed": units_completed,
        "artifacts": dict(artifacts or {}),
    }


def build_optional_task_dict(
    name: str,
    *,
    status: str = "pending",
    progress: float = 0.0,
    detail: str = "",
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build an optional-task payload compatible with ``RunSnapshot.from_dict``."""
    return {
        "name": name,
        "status": status,
        "progress": progress,
        "detail": detail,
        "artifacts": dict(artifacts or {}),
    }


def build_snapshot_dict(
    *,
    run_id: str = "run-1",
    input_fingerprint: str = "",
    params_fingerprint: str = "",
    status: str = "pending",
    target_stage: str = "network",
    current_stage: str = "",
    overall_progress: float = 0.0,
    stages: dict[str, dict[str, Any]] | None = None,
    optional_tasks: dict[str, dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
    errors: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    last_event: str = "",
) -> dict[str, Any]:
    """Build a run snapshot payload compatible with ``RunSnapshot.from_dict``."""
    return {
        "run_id": run_id,
        "input_fingerprint": input_fingerprint,
        "params_fingerprint": params_fingerprint,
        "status": status,
        "target_stage": target_stage,
        "current_stage": current_stage,
        "overall_progress": overall_progress,
        "stages": dict(stages or {}),
        "optional_tasks": dict(optional_tasks or {}),
        "artifacts": dict(artifacts or {}),
        "errors": list(errors or []),
        "provenance": dict(provenance or {}),
        "last_event": last_event,
    }


def build_run_context(
    run_dir: Path,
    *,
    input_fingerprint: str = "input-a",
    params_fingerprint: str = "params-a",
    target_stage: str = "network",
    provenance: dict[str, Any] | None = None,
) -> RunContext:
    """Build a ``RunContext`` with standard test defaults."""
    return RunContext(
        run_dir=run_dir,
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
        target_stage=target_stage,
        provenance=provenance or {"source": "test-builder"},
    )


def materialize_run_snapshot(run_dir: Path, snapshot: dict[str, Any]) -> Path:
    """Write a structured run snapshot under ``99_Metadata``."""
    metadata_dir = run_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = metadata_dir / "run_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot_path


def materialize_checkpoint_surface(
    root: Path,
    *,
    stages: Iterable[str],
    payloads: dict[str, Any] | None = None,
    structured: bool = True,
) -> Path:
    """Write checkpoint files for the requested stages."""
    checkpoint_dir = root / "02_Output" / "python_results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        joblib.dump(
            (payloads or {}).get(stage, {stage: True}),
            checkpoint_dir / f"checkpoint_{stage}.pkl",
        )
    return checkpoint_dir


__all__ = [
    "build_optional_task_dict",
    "build_run_context",
    "build_snapshot_dict",
    "build_stage_snapshot_dict",
    "materialize_checkpoint_surface",
    "materialize_run_snapshot",
]


