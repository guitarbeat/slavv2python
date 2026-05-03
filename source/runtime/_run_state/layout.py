"""Helpers for resolving run-state filesystem layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Resolved filesystem layout for a structured run."""

    run_root: Path
    refs_dir: Path
    params_dir: Path
    artifacts_dir: Path
    analysis_dir: Path
    metadata_dir: Path
    stages_dir: Path
    checkpoints_dir: Path
    normalized_dir: Path
    hashes_dir: Path

    @property
    def snapshot_path(self) -> Path:
        return self.metadata_dir / "run_snapshot.json"

    @property
    def manifest_path(self) -> Path:
        return self.metadata_dir / "run_manifest.json"

    def checkpoint_path(self, stage: str) -> Path:
        return self.checkpoints_dir / f"checkpoint_{stage}.pkl"

    def stage_dir(self, stage: str) -> Path:
        path = self.stages_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_directories(self) -> None:
        """Create the on-disk directories needed by the run layout."""
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.hashes_dir.mkdir(parents=True, exist_ok=True)


def resolve_run_layout(
        *,
        run_dir: str | Path | None,
) -> RunLayout:
    """Resolve a structured filesystem layout for run-state bookkeeping."""
    if run_dir is None:
        raise ValueError("run_dir is required for run state")
    run_root = Path(run_dir)
    artifacts_dir = run_root / "02_Output" / "python_results"
    analysis_dir = run_root / "03_Analysis"
    return RunLayout(
        run_root=run_root,
        refs_dir=run_root / "00_Refs",
        params_dir=run_root / "01_Params",
        artifacts_dir=artifacts_dir,
        analysis_dir=analysis_dir,
        metadata_dir=run_root / "99_Metadata",
        stages_dir=artifacts_dir / "stages",
        checkpoints_dir=artifacts_dir / "checkpoints",
        normalized_dir=analysis_dir / "normalized",
        hashes_dir=analysis_dir / "hashes",
    )


__all__ = ["RunLayout", "resolve_run_layout"]
