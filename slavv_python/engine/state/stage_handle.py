"""Per-stage handle for checkpoints, artifacts, and progress."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, cast

from slavv_python.engine.state.io import atomic_joblib_dump, atomic_write_json
from slavv_python.engine.state.models import _now_iso
from slavv_python.utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from slavv_python.engine.state.run_ledger import RunContext


class StageController:
    """Stage-specific helper for persisted state and artifacts."""

    def __init__(self, run_context: RunContext, name: str):
        """Initialize StageController.

        Args:
            run_context: The parent run context.
            name: The name of the stage (e.g., 'energy', 'vertices').
        """
        self.run_context = run_context
        self.name = name

    @property
    def stage_dir(self) -> Path:
        """Get the directory path for this stage's artifacts."""
        return self.run_context.layout.stage_dir(self.name)

    @property
    def checkpoint_path(self) -> Path:
        """Get the path to the main checkpoint file for this stage."""
        return self.run_context.layout.checkpoint_path(self.name)

    @property
    def manifest_path(self) -> Path:
        """Get the path to the stage manifest file."""
        return self.stage_dir / "stage_manifest.json"

    @property
    def state_path(self) -> Path:
        """Get the path to the resume state file."""
        return self.stage_dir / "resume_state.json"

    def artifact_path(self, file_name: str) -> Path:
        """Get the path to a specific artifact file within the stage directory.

        Args:
            file_name: The name of the artifact file.

        Returns:
            Path: The full path to the artifact.
        """
        return self.stage_dir / file_name

    def load_state(self) -> dict[str, Any]:
        """Load the resumable state for this stage.

        Returns:
            dict[str, Any]: The loaded state dictionary, or an empty dict if it doesn't exist.

        Raises:
            ValueError: If the state file exists but does not contain a JSON object.
        """
        if not self.state_path.exists():
            return {}
        with open(self.state_path, encoding="utf-8") as handle:
            loaded_state = json.load(handle)
        if not isinstance(loaded_state, dict):
            raise ValueError(f"Expected JSON object in {self.state_path}")
        return cast("dict[str, Any]", loaded_state)

    def save_state(self, state: dict[str, Any]) -> None:
        """Save the resumable state for this stage.

        Args:
            state: The state dictionary to persist.
        """
        atomic_write_json(self.state_path, state)

    def remove_state(self) -> None:
        """Delete the resume state file for this stage."""
        if self.state_path.exists():
            self.state_path.unlink()

    def load_checkpoint(self) -> Any:
        """Load the main checkpoint payload for this stage.

        Returns:
            Any: The deserialized checkpoint data.
        """
        return safe_load(self.checkpoint_path)

    def save_checkpoint(self, data: Any) -> None:
        """Save the main checkpoint payload for this stage.

        Args:
            data: The payload to persist (must be joblib-serializable).
        """
        atomic_joblib_dump(data, self.checkpoint_path)

    def begin(
        self,
        *,
        detail: str = "",
        units_total: int = 0,
        units_completed: int = 0,
        substage: str = "",
        resumed: bool = False,
    ) -> None:
        """Mark the beginning of stage execution.

        Args:
            detail: Friendly description of the work being started.
            units_total: Total number of work units for progress tracking.
            units_completed: Initial number of completed units.
            substage: Optional internal substage name.
            resumed: Whether this execution is resuming from a previous attempt.
        """
        self.run_context.begin_stage(
            self.name,
            detail=detail,
            units_total=units_total,
            units_completed=units_completed,
            substage=substage,
            resumed=resumed,
        )

    def update(
        self,
        *,
        detail: str | None = None,
        units_total: int | None = None,
        units_completed: int | None = None,
        progress: float | None = None,
        substage: str | None = None,
        resumed: bool | None = None,
    ) -> None:
        """Update the execution progress for this stage.

        Args:
            detail: Updated description of current work.
            units_total: Updated total work units.
            units_completed: Updated completed work units.
            progress: Explicit progress fraction (0.0 to 1.0).
            substage: Updated internal substage name.
            resumed: Updated resumption status.
        """
        state = self.load_state()
        cursor = dict(state.get("progress_cursor", {}))
        if units_total is not None:
            cursor["units_total"] = units_total
        if units_completed is not None:
            cursor["units_completed"] = units_completed
        if progress is not None:
            cursor["progress"] = progress
        if substage is not None:
            cursor["substage"] = substage
        state["progress_cursor"] = cursor
        state["heartbeat_at"] = _now_iso()
        self.save_state(state)
        self.run_context.update_stage(
            self.name,
            detail=detail,
            units_total=units_total,
            units_completed=units_completed,
            progress=progress,
            substage=substage,
            resumed=resumed,
        )

    def complete(
        self,
        *,
        detail: str = "",
        artifacts: dict[str, str] | None = None,
        resumed: bool | None = None,
    ) -> None:
        """Mark the stage as successfully completed.

        Args:
            detail: Friendly description of the completed work.
            artifacts: Dictionary of additional artifacts created by this stage.
            resumed: Whether the successful execution was a resumed one.
        """
        manifest = {
            "stage": self.name,
            "checkpoint": str(self.checkpoint_path),
            "artifacts": artifacts or {},
            "completed_at": _now_iso(),
        }
        atomic_write_json(self.manifest_path, manifest)
        self.run_context.complete_stage(
            self.name,
            detail=detail,
            artifacts={"checkpoint": str(self.checkpoint_path), **(artifacts or {})},
            resumed=resumed,
        )
        self.remove_state()


__all__ = ["StageController"]
