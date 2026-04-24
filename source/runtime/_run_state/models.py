from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import STATUS_PENDING


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class StageSnapshot:
    """Serializable state for a pipeline stage."""

    name: str
    status: str = STATUS_PENDING
    progress: float = 0.0
    resumed: bool = False
    detail: str = ""
    substage: str = ""
    units_total: int = 0
    units_completed: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None


@dataclass
class TaskSnapshot:
    """Serializable state for optional work attached to a run."""

    name: str
    status: str = STATUS_PENDING
    progress: float = 0.0
    detail: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class RunSnapshot:
    """Serializable snapshot for a resumable run."""

    run_id: str
    input_fingerprint: str = ""
    params_fingerprint: str = ""
    status: str = STATUS_PENDING
    target_stage: str = "network"
    current_stage: str = ""
    overall_progress: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    stages: dict[str, StageSnapshot] = field(default_factory=dict)
    optional_tasks: dict[str, TaskSnapshot] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_event: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSnapshot:
        stage_data = data.get("stages", {})
        task_data = data.get("optional_tasks", {})
        stages = {
            name: StageSnapshot(
                **dict({"name": name}, **{k: v for k, v in values.items() if k != "name"})
            )
            for name, values in stage_data.items()
        }
        tasks = {
            name: TaskSnapshot(
                **dict({"name": name}, **{k: v for k, v in values.items() if k != "name"})
            )
            for name, values in task_data.items()
        }
        return cls(
            run_id=data.get("run_id", uuid.uuid4().hex[:12]),
            input_fingerprint=data.get("input_fingerprint", ""),
            params_fingerprint=data.get("params_fingerprint", ""),
            status=data.get("status", STATUS_PENDING),
            target_stage=data.get("target_stage", "network"),
            current_stage=data.get("current_stage", ""),
            overall_progress=float(data.get("overall_progress", 0.0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            eta_seconds=data.get("eta_seconds"),
            stages=stages,
            optional_tasks=tasks,
            artifacts=data.get("artifacts", {}),
            errors=data.get("errors", []),
            provenance=data.get("provenance", {}),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            last_event=data.get("last_event", ""),
        )


@dataclass
class ProgressEvent:
    """Progress payload emitted by the pipeline."""

    stage: str
    status: str
    overall_progress: float
    stage_progress: float
    detail: str
    resumed: bool
    snapshot: RunSnapshot
