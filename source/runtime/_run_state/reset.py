"""Helpers for clearing persisted state when a pipeline stage is reset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import PIPELINE_STAGES
from .models import StageSnapshot

if TYPE_CHECKING:
    from pathlib import Path

    from .context import StageController


def remove_stage_dir_contents(stage_dir: Path) -> None:
    """Remove all files and nested directories inside a stage directory."""
    if not stage_dir.exists():
        return
    for child in stage_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_dir():
                    nested.rmdir()
            child.rmdir()


def clear_stage_runtime_artifacts(controller: StageController) -> None:
    """Remove checkpoint, manifest, resume state, and stage-directory contents."""
    if controller.checkpoint_path.exists():
        controller.checkpoint_path.unlink()
    if controller.manifest_path.exists():
        controller.manifest_path.unlink()
    if controller.state_path.exists():
        controller.state_path.unlink()
    remove_stage_dir_contents(controller.stage_dir)


def reset_stage_snapshots(
        stages: dict[str, StageSnapshot],
        *,
        start_stage: str,
) -> list[str]:
    """Reset stage snapshots from the requested stage onward and return affected stages."""
    if start_stage not in PIPELINE_STAGES:
        return []
    start_index = PIPELINE_STAGES.index(start_stage)
    affected = PIPELINE_STAGES[start_index:]
    for stage in affected:
        stages[stage] = StageSnapshot(name=stage)
    return affected


__all__ = [
    "clear_stage_runtime_artifacts",
    "remove_stage_dir_contents",
    "reset_stage_snapshots",
]
