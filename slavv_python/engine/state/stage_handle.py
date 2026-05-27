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
        self.run_context = run_context
        self.name = name

    @property
    def stage_dir(self) -> Path:
        return self.run_context.layout.stage_dir(self.name)

    @property
    def checkpoint_path(self) -> Path:
        return self.run_context.layout.checkpoint_path(self.name)

    @property
    def manifest_path(self) -> Path:
        return self.stage_dir / "stage_manifest.json"

    @property
    def state_path(self) -> Path:
        return self.stage_dir / "resume_state.json"

    def artifact_path(self, file_name: str) -> Path:
        return self.stage_dir / file_name

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        with open(self.state_path, encoding="utf-8") as handle:
            loaded_state = json.load(handle)
        if not isinstance(loaded_state, dict):
            raise ValueError(f"Expected JSON object in {self.state_path}")
        return cast("dict[str, Any]", loaded_state)

    def save_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    def remove_state(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    def load_checkpoint(self) -> Any:
        return safe_load(self.checkpoint_path)

    def save_checkpoint(self, data: Any) -> None:
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
