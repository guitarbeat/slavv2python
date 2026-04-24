"""Helpers for resolving run-state filesystem layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Resolved filesystem layout for a structured run."""

    run_root: Path
    artifacts_dir: Path
    metadata_dir: Path
    stages_dir: Path
    checkpoints_dir: Path

    @property
    def snapshot_path(self) -> Path:
        return self.metadata_dir / "run_snapshot.json"

    def checkpoint_path(self, stage: str) -> Path:
        return self.checkpoints_dir / f"checkpoint_{stage}.pkl"

    def stage_dir(self, stage: str) -> Path:
        path = self.stages_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_directories(self) -> None:
        """Create the on-disk directories needed by the run layout."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


def resolve_run_layout(
    *,
    run_dir: str | Path | None,
) -> RunLayout:
    """Resolve a structured filesystem layout for run-state bookkeeping."""
    if run_dir is None:
        raise ValueError("run_dir is required for run state")
    run_root = Path(run_dir)
    artifacts_dir = run_root / "02_Output" / "python_results"
    return RunLayout(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        metadata_dir=run_root / "99_Metadata",
        stages_dir=artifacts_dir / "stages",
        checkpoints_dir=artifacts_dir / "checkpoints",
    )


__all__ = ["RunLayout", "resolve_run_layout"]
