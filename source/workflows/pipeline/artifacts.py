"""Helpers for common stage checkpoint lifecycle handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def stage_artifacts(stage_controller: Any) -> dict[str, str]:
    """Collect persisted stage artifacts excluding resume metadata."""
    artifacts: dict[str, str] = {}
    stage_dir = Path(stage_controller.stage_dir)
    if stage_dir.exists():
        for artifact in stage_dir.iterdir():
            if artifact.name == "resume_state.json":
                continue
            if artifact.is_file() or artifact.is_dir():
                artifacts[artifact.name] = str(artifact)
    return artifacts


def load_cached_stage_result(stage_controller: Any, *, detail: str) -> dict[str, Any]:
    """Load a cached stage payload and mark the stage resumed."""
    payload = cast("dict[str, Any]", stage_controller.load_checkpoint())
    stage_controller.complete(
        detail=detail,
        artifacts=stage_artifacts(stage_controller),
        resumed=True,
    )
    return payload


def persist_stage_result(
    stage_controller: Any,
    payload: dict[str, Any],
    *,
    detail: str,
) -> dict[str, Any]:
    """Persist a stage payload and mark the stage complete."""
    stage_controller.save_checkpoint(payload)
    stage_controller.complete(
        detail=detail,
        artifacts=stage_artifacts(stage_controller),
    )
    return payload


def resolve_resumable_stage(
    stage_controller: Any,
    *,
    force_rerun: bool,
    cached_log_label: str,
    cached_detail: str,
    success_detail: str,
    compute_fn,
    logger: Any,
) -> dict[str, Any]:
    """Load a cached stage or compute and persist a fresh stage payload."""
    if stage_controller.checkpoint_path.exists() and not force_rerun:
        logger.info("Loading cached %s from %s", cached_log_label, stage_controller.checkpoint_path)
        return load_cached_stage_result(stage_controller, detail=cached_detail)

    payload = cast("dict[str, Any]", compute_fn())
    return persist_stage_result(stage_controller, payload, detail=success_detail)


__all__ = [
    "load_cached_stage_result",
    "persist_stage_result",
    "resolve_resumable_stage",
    "stage_artifacts",
]
