"""File-backed run state for resumable SLAVV processing."""

from __future__ import annotations

import calendar
import copy
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import joblib
import numpy as np

from slavv.utils.safe_unpickle import safe_load

logger = logging.getLogger(__name__)

PREPROCESS_STAGE = "preprocess"
PIPELINE_STAGES = ["energy", "vertices", "edges", "network"]
TRACKED_RUN_STAGES = [PREPROCESS_STAGE, *PIPELINE_STAGES]
STAGE_WEIGHTS = {
    PREPROCESS_STAGE: 0.05,
    "energy": 0.35,
    "vertices": 0.15,
    "edges": 0.30,
    "network": 0.15,
}
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_TARGET = "completed_target"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "resume_blocked"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return [_normalize_for_json(v) for v in sorted(value)]
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize a value with deterministic ordering."""
    return json.dumps(_normalize_for_json(value), sort_keys=True, separators=(",", ":"))


def fingerprint_jsonable(value: Any) -> str:
    """Create a content hash for JSON-like data."""
    payload = stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_array(array: np.ndarray) -> str:
    """Create a content hash for a numpy array."""
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def fingerprint_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Create a content hash for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            if chunk := handle.read(chunk_size):
                hasher.update(chunk)
            else:
                break
    return hasher.hexdigest()


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Atomically write JSON content."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_normalize_for_json(data), handle, indent=2, sort_keys=True)
        _replace_with_retry(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_joblib_dump(value: Any, path: str | Path) -> None:
    """Atomically write a joblib artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(value, tmp_name)
        _replace_with_retry(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _replace_with_retry(
    tmp_name: str, target: Path, *, attempts: int = 20, delay: float = 0.25
) -> None:
    """Retry atomic replacement to tolerate transient Windows file locks."""
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(tmp_name, target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    if last_error is not None:
        raise last_error


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


def load_run_snapshot(path_or_dir: str | Path) -> RunSnapshot | None:
    """Load a run snapshot from a file or directory if present."""
    path = Path(path_or_dir)
    candidates = []
    if path.is_dir():
        candidates.extend(
            [
                path / "run_snapshot.json",
                path / "99_Metadata" / "run_snapshot.json",
                path / "metadata" / "run_snapshot.json",
            ]
        )
    else:
        candidates.append(path)

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as handle:
                return RunSnapshot.from_dict(json.load(handle))
    return None


def load_legacy_run_snapshot(
    checkpoint_dir: str | Path, *, target_stage: str = "network"
) -> RunSnapshot | None:
    """Inspect legacy checkpoint directories without mutating them."""
    checkpoints_dir = Path(checkpoint_dir)
    checkpoint_paths = {
        stage: checkpoints_dir / f"checkpoint_{stage}.pkl" for stage in PIPELINE_STAGES
    }
    if not any(path.exists() for path in checkpoint_paths.values()):
        return None

    snapshot = RunSnapshot(
        run_id=hashlib.sha1(str(checkpoints_dir.resolve()).encode("utf-8")).hexdigest()[:12],
        target_stage=target_stage,
        status=STATUS_PENDING,
        stages=_ensure_stage_map(),
        provenance={"layout": "legacy"},
    )
    preprocess_stage = snapshot.stages[PREPROCESS_STAGE]
    preprocess_stage.status = STATUS_COMPLETED
    preprocess_stage.progress = 1.0
    preprocess_stage.units_total = 1
    preprocess_stage.units_completed = 1
    preprocess_stage.resumed = True
    for stage, path in checkpoint_paths.items():
        if not path.exists():
            continue
        stage_snapshot = snapshot.stages[stage]
        stage_snapshot.status = STATUS_COMPLETED
        stage_snapshot.progress = 1.0
        stage_snapshot.units_total = 1
        stage_snapshot.units_completed = 1
        stage_snapshot.resumed = True
        stage_snapshot.artifacts["checkpoint"] = str(path)
        stage_snapshot.completed_at = _now_iso()
    total = STAGE_WEIGHTS[PREPROCESS_STAGE] + sum(STAGE_WEIGHTS[stage] for stage in PIPELINE_STAGES)
    snapshot.overall_progress = (
        sum(STAGE_WEIGHTS[stage] * snapshot.stages[stage].progress for stage in TRACKED_RUN_STAGES)
        / total
    )
    return snapshot


def _ensure_stage_map(existing: dict[str, StageSnapshot] | None = None) -> dict[str, StageSnapshot]:
    stages = {name: StageSnapshot(name=name) for name in TRACKED_RUN_STAGES}
    if existing:
        stages |= existing
        for name in TRACKED_RUN_STAGES:
            stages.setdefault(name, StageSnapshot(name=name))
    return stages


class StageController:
    """Stage-specific helper for persisted state and artifacts."""

    def __init__(self, run_context: RunContext, name: str):
        self.run_context = run_context
        self.name = name

    @property
    def stage_dir(self) -> Path:
        path = self.run_context.stages_dir / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_path(self) -> Path:
        return self.run_context.checkpoint_path(self.name)

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


class RunContext:
    """Shared run ledger and file layout for resumable processing."""

    def __init__(
        self,
        *,
        run_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        input_fingerprint: str = "",
        params_fingerprint: str = "",
        target_stage: str | None = "network",
        provenance: dict[str, Any] | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
        legacy: bool = False,
    ):
        self.legacy = legacy or (checkpoint_dir is not None and run_dir is None)
        if self.legacy:
            if checkpoint_dir is None:
                raise ValueError("checkpoint_dir is required for legacy run state")
            self.run_root = Path(checkpoint_dir)
            self.artifacts_dir = self.run_root
            self.metadata_dir = self.run_root
            self.stages_dir = self.run_root / "stage_state"
            self.checkpoints_dir = self.run_root
        else:
            if run_dir is None:
                raise ValueError("run_dir is required for structured run state")
            self.run_root = Path(run_dir)
            self.metadata_dir = self.run_root / "99_Metadata"
            self.artifacts_dir = self.run_root / "02_Output" / "python_results"
            self.stages_dir = self.artifacts_dir / "stages"
            self.checkpoints_dir = self.artifacts_dir / "checkpoints"

        self.snapshot_path = self.metadata_dir / "run_snapshot.json"
        self.event_callback = event_callback
        self.start_time = time.time()
        self.snapshot = self._load_or_create_snapshot(
            input_fingerprint=input_fingerprint,
            params_fingerprint=params_fingerprint,
            target_stage=target_stage,
            provenance=provenance or {},
        )
        self.persist()

    @classmethod
    def from_existing(
        cls,
        run_dir: str | Path,
        *,
        legacy: bool = False,
        checkpoint_dir: str | Path | None = None,
        target_stage: str | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> RunContext:
        return cls(
            run_dir=None if legacy else run_dir,
            checkpoint_dir=(checkpoint_dir or run_dir) if legacy else None,
            target_stage=target_stage,
            event_callback=event_callback,
            legacy=legacy,
        )

    def _load_or_create_snapshot(
        self,
        *,
        input_fingerprint: str,
        params_fingerprint: str,
        target_stage: str | None,
        provenance: dict[str, Any],
    ) -> RunSnapshot:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        existing = load_run_snapshot(self.snapshot_path)
        if existing is None and self.legacy:
            existing = self._bootstrap_legacy_snapshot(target_stage=target_stage or "network")

        if existing is not None:
            existing.stages = _ensure_stage_map(existing.stages)
            # Preserve stored fingerprints until the explicit resume guard compares
            # them against the incoming request. Overwriting them here would allow
            # stale checkpoints to slip past mismatch detection.
            if input_fingerprint and not existing.input_fingerprint:
                existing.input_fingerprint = input_fingerprint
            if params_fingerprint and not existing.params_fingerprint:
                existing.params_fingerprint = params_fingerprint
            if target_stage is not None:
                existing.target_stage = target_stage
            if provenance:
                existing.provenance.update(_normalize_for_json(provenance))
            existing.updated_at = _now_iso()
            return existing

        return RunSnapshot(
            run_id=uuid.uuid4().hex[:12],
            input_fingerprint=input_fingerprint,
            params_fingerprint=params_fingerprint,
            status=STATUS_PENDING,
            target_stage=target_stage or "network",
            stages=_ensure_stage_map(),
            provenance=_normalize_for_json(
                {
                    "layout": "legacy" if self.legacy else "structured",
                    **provenance,
                }
            ),
        )

    def _bootstrap_legacy_snapshot(self, *, target_stage: str) -> RunSnapshot | None:
        snapshot = load_legacy_run_snapshot(self.checkpoints_dir, target_stage=target_stage)
        if snapshot is None:
            return None
        snapshot.overall_progress = self._calculate_overall_progress(
            snapshot.stages,
            preprocess_done=bool(snapshot.artifacts.get("preprocess_done")),
        )
        return snapshot

    def checkpoint_path(self, stage: str) -> Path:
        return self.checkpoints_dir / f"checkpoint_{stage}.pkl"

    def stage(self, name: str) -> StageController:
        if name not in PIPELINE_STAGES:
            valid = ", ".join(PIPELINE_STAGES)
            raise ValueError(f"stage must be one of: {valid}")
        return StageController(self, name)

    def persist(self) -> None:
        self.snapshot.elapsed_seconds = max(
            self.snapshot.elapsed_seconds,
            time.time() - self.start_time,
        )
        self.snapshot.updated_at = _now_iso()
        atomic_write_json(self.snapshot_path, self.snapshot.to_dict())

    def ensure_resume_allowed(
        self,
        *,
        input_fingerprint: str,
        params_fingerprint: str,
        force_rerun_from: str | None = None,
    ) -> None:
        mismatch = []
        if self.snapshot.input_fingerprint and self.snapshot.input_fingerprint != input_fingerprint:
            mismatch.append("input")
        if (
            self.snapshot.params_fingerprint
            and self.snapshot.params_fingerprint != params_fingerprint
        ):
            mismatch.append("parameters")

        if not mismatch:
            self.snapshot.input_fingerprint = input_fingerprint
            self.snapshot.params_fingerprint = params_fingerprint
            self.persist()
            return

        if force_rerun_from == "energy":
            logger.info(
                "Resuming with explicit rerun from energy after %s change",
                ", ".join(mismatch),
            )
            self.reset_pipeline_state()
            self.snapshot.input_fingerprint = input_fingerprint
            self.snapshot.params_fingerprint = params_fingerprint
            self.snapshot.status = STATUS_PENDING
            self.snapshot.last_event = "Pipeline reset after resume guard mismatch"
            self.persist()
            return

        message = (
            "Resume blocked because the "
            + " and ".join(mismatch)
            + " fingerprint changed. Re-run with force_rerun_from='energy' to start a fresh pipeline."
        )
        self.snapshot.status = STATUS_BLOCKED
        self.snapshot.last_event = message
        self.snapshot.errors.append({"message": message, "at": _now_iso()})
        self.persist()
        raise RuntimeError(message)

    def reset_pipeline_state(self) -> None:
        self.reset_pipeline_state_from("energy")

    def reset_pipeline_state_from(self, start_stage: str) -> None:
        if start_stage not in PIPELINE_STAGES:
            return
        start_index = PIPELINE_STAGES.index(start_stage)
        for stage in PIPELINE_STAGES[start_index:]:
            controller = self.stage(stage)
            if controller.checkpoint_path.exists():
                controller.checkpoint_path.unlink()
            if controller.manifest_path.exists():
                controller.manifest_path.unlink()
            if controller.state_path.exists():
                controller.state_path.unlink()
            if controller.stage_dir.exists():
                for child in controller.stage_dir.iterdir():
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
            self.snapshot.stages[stage] = StageSnapshot(name=stage)
        self.persist()

    def mark_preprocess_complete(self) -> None:
        stage_snapshot = self.snapshot.stages.setdefault(
            PREPROCESS_STAGE,
            StageSnapshot(name=PREPROCESS_STAGE),
        )
        if stage_snapshot.started_at is None:
            stage_snapshot.started_at = _now_iso()
        stage_snapshot.status = STATUS_COMPLETED
        stage_snapshot.progress = 1.0
        stage_snapshot.units_total = max(stage_snapshot.units_total, 1)
        stage_snapshot.units_completed = stage_snapshot.units_total
        stage_snapshot.detail = "Preprocessing complete"
        stage_snapshot.updated_at = _now_iso()
        stage_snapshot.completed_at = _now_iso()
        self.snapshot.artifacts["preprocess_done"] = "true"
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
        self.snapshot.last_event = "Preprocessing complete"
        self.persist()
        self.emit_event(PREPROCESS_STAGE, STATUS_COMPLETED, detail="Preprocessing complete")

    def mark_run_status(self, status: str, *, current_stage: str = "", detail: str = "") -> None:
        self.snapshot.status = status
        if current_stage:
            self.snapshot.current_stage = current_stage
        if detail:
            self.snapshot.last_event = detail
        self.persist()
        self.emit_event(current_stage or self.snapshot.current_stage, status, detail=detail)

    def begin_stage(
        self,
        stage: str,
        *,
        detail: str = "",
        units_total: int = 0,
        units_completed: int = 0,
        substage: str = "",
        resumed: bool = False,
    ) -> None:
        stage_snapshot = self.snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
        if stage_snapshot.started_at is None:
            stage_snapshot.started_at = _now_iso()
        stage_snapshot.status = STATUS_RUNNING
        stage_snapshot.updated_at = _now_iso()
        stage_snapshot.detail = detail
        stage_snapshot.substage = substage
        stage_snapshot.units_total = units_total or stage_snapshot.units_total
        stage_snapshot.units_completed = units_completed
        stage_snapshot.resumed = resumed or stage_snapshot.resumed
        if stage_snapshot.units_total > 0:
            stage_snapshot.progress = min(
                1.0,
                stage_snapshot.units_completed / stage_snapshot.units_total,
            )
        self.snapshot.current_stage = stage
        self.snapshot.status = STATUS_RUNNING
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
        self.snapshot.last_event = detail or f"Running {stage}"
        self.persist()
        self.emit_event(stage, STATUS_RUNNING, detail=detail)

    def update_stage(
        self,
        stage: str,
        *,
        detail: str | None = None,
        units_total: int | None = None,
        units_completed: int | None = None,
        progress: float | None = None,
        substage: str | None = None,
        resumed: bool | None = None,
    ) -> None:
        stage_snapshot = self.snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
        stage_snapshot.status = STATUS_RUNNING
        stage_snapshot.updated_at = _now_iso()
        if detail is not None:
            stage_snapshot.detail = detail
        if substage is not None:
            stage_snapshot.substage = substage
        if units_total is not None:
            stage_snapshot.units_total = units_total
        if units_completed is not None:
            stage_snapshot.units_completed = units_completed
        if resumed is not None:
            stage_snapshot.resumed = resumed
        if progress is not None:
            stage_snapshot.progress = max(0.0, min(1.0, progress))
        elif stage_snapshot.units_total > 0:
            stage_snapshot.progress = min(
                1.0,
                stage_snapshot.units_completed / stage_snapshot.units_total,
            )
        stage_snapshot.eta_seconds = self._estimate_eta(stage_snapshot)
        self.snapshot.current_stage = stage
        self.snapshot.status = STATUS_RUNNING
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
        self.snapshot.eta_seconds = self._estimate_run_eta(self.snapshot)
        self.snapshot.last_event = stage_snapshot.detail or f"Running {stage}"
        self.persist()
        self.emit_event(stage, STATUS_RUNNING, detail=stage_snapshot.detail)

    def complete_stage(
        self,
        stage: str,
        *,
        detail: str = "",
        artifacts: dict[str, str] | None = None,
        resumed: bool | None = None,
    ) -> None:
        stage_snapshot = self.snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
        stage_snapshot.status = STATUS_COMPLETED
        stage_snapshot.progress = 1.0
        stage_snapshot.updated_at = _now_iso()
        stage_snapshot.completed_at = _now_iso()
        stage_snapshot.units_total = max(
            stage_snapshot.units_total,
            stage_snapshot.units_completed,
            1,
        )
        stage_snapshot.units_completed = stage_snapshot.units_total
        if detail:
            stage_snapshot.detail = detail
        if artifacts:
            stage_snapshot.artifacts.update(artifacts)
            self.snapshot.artifacts.update({f"{stage}.{k}": v for k, v in artifacts.items()})
        if resumed is not None:
            stage_snapshot.resumed = resumed
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
        self.snapshot.eta_seconds = self._estimate_run_eta(self.snapshot)
        self.snapshot.last_event = detail or f"Completed {stage}"
        self.persist()
        self.emit_event(stage, STATUS_COMPLETED, detail=stage_snapshot.detail)

    def fail_stage(self, stage: str, error: BaseException | str) -> None:
        message = str(error)
        stage_snapshot = self.snapshot.stages.setdefault(stage, StageSnapshot(name=stage))
        stage_snapshot.status = STATUS_FAILED
        stage_snapshot.updated_at = _now_iso()
        stage_snapshot.detail = message
        self.snapshot.status = STATUS_FAILED
        self.snapshot.current_stage = stage
        self.snapshot.last_event = message
        self.snapshot.errors.append({"stage": stage, "message": message, "at": _now_iso()})
        self.persist()
        self.emit_event(stage, STATUS_FAILED, detail=message)

    def update_optional_task(
        self,
        name: str,
        *,
        status: str,
        detail: str = "",
        progress: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        task = self.snapshot.optional_tasks.setdefault(name, TaskSnapshot(name=name))
        if task.started_at is None and status == STATUS_RUNNING:
            task.started_at = _now_iso()
        task.status = status
        task.updated_at = _now_iso()
        task.detail = detail
        if progress is not None:
            task.progress = progress
        if status == STATUS_COMPLETED:
            task.progress = 1.0
            task.completed_at = _now_iso()
        if artifacts:
            task.artifacts.update(artifacts)
        self.snapshot.last_event = detail or f"Task {name}: {status}"
        self.persist()

    def finalize_run(self, *, stop_after: str | None = None) -> None:
        if stop_after and stop_after != "network":
            self.snapshot.status = STATUS_COMPLETED_TARGET
        else:
            self.snapshot.status = STATUS_COMPLETED
            self.snapshot.overall_progress = 1.0
        self.snapshot.current_stage = stop_after or "network"
        self.snapshot.eta_seconds = 0.0
        self.snapshot.last_event = "Run completed"
        self.persist()
        self.emit_event(self.snapshot.current_stage, self.snapshot.status, detail="Run completed")

    def emit_event(self, stage: str, status: str, *, detail: str = "") -> None:
        if self.event_callback is None:
            return
        stage_snapshot = self.snapshot.stages.get(stage, StageSnapshot(name=stage))
        payload = ProgressEvent(
            stage=stage,
            status=status,
            overall_progress=self.snapshot.overall_progress,
            stage_progress=stage_snapshot.progress,
            detail=detail,
            resumed=stage_snapshot.resumed,
            snapshot=copy.deepcopy(self.snapshot),
        )
        self.event_callback(payload)

    def _estimate_eta(self, stage_snapshot: StageSnapshot) -> float | None:
        if stage_snapshot.progress <= 0.0 or stage_snapshot.started_at is None:
            return None
        started = self._parse_time(stage_snapshot.started_at)
        if started is None:
            return None
        elapsed = max(0.0, time.time() - started)
        remaining = elapsed * (1.0 - stage_snapshot.progress) / stage_snapshot.progress
        stage_snapshot.elapsed_seconds = elapsed
        return remaining

    def _estimate_run_eta(self, snapshot: RunSnapshot) -> float | None:
        if snapshot.overall_progress <= 0.0:
            return None
        created = self._parse_time(snapshot.created_at)
        if created is None:
            return None
        elapsed = max(0.0, time.time() - created)
        snapshot.elapsed_seconds = elapsed
        return elapsed * (1.0 - snapshot.overall_progress) / snapshot.overall_progress

    def _calculate_overall_progress(
        self,
        stages: dict[str, StageSnapshot],
        preprocess_done: bool | None = None,
    ) -> float:
        total = STAGE_WEIGHTS[PREPROCESS_STAGE] + sum(
            STAGE_WEIGHTS[stage] for stage in PIPELINE_STAGES
        )
        progress = 0.0
        if preprocess_done is None:
            snapshot = getattr(self, "snapshot", None)
            preprocess_stage = stages.get(PREPROCESS_STAGE, StageSnapshot(name=PREPROCESS_STAGE))
            preprocess_done = bool(snapshot and snapshot.artifacts.get("preprocess_done")) or (
                preprocess_stage.status == STATUS_COMPLETED
            )
        if preprocess_done:
            progress += STAGE_WEIGHTS[PREPROCESS_STAGE]
        for stage_name in PIPELINE_STAGES:
            progress += (
                STAGE_WEIGHTS[stage_name]
                * stages.get(
                    stage_name,
                    StageSnapshot(stage_name),
                ).progress
            )
        return max(0.0, min(1.0, progress / total))

    @staticmethod
    def _parse_time(timestamp: str | None) -> float | None:
        if not timestamp:
            return None
        try:
            return calendar.timegm(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return None


def target_stage_progress(snapshot: RunSnapshot) -> float:
    """Return progress toward the user-selected pipeline target."""
    if snapshot.target_stage not in PIPELINE_STAGES:
        return snapshot.overall_progress
    index = PIPELINE_STAGES.index(snapshot.target_stage)
    selected = PIPELINE_STAGES[: index + 1]
    total = STAGE_WEIGHTS[PREPROCESS_STAGE] + sum(STAGE_WEIGHTS[stage] for stage in selected)
    preprocess_stage = snapshot.stages.get(PREPROCESS_STAGE, StageSnapshot(name=PREPROCESS_STAGE))
    preprocess_done = bool(snapshot.artifacts.get("preprocess_done")) or (
        preprocess_stage.status == STATUS_COMPLETED
    )
    progress = STAGE_WEIGHTS[PREPROCESS_STAGE] if preprocess_done else 0.0
    for stage in selected:
        progress += (
            STAGE_WEIGHTS[stage]
            * snapshot.stages.get(
                stage,
                StageSnapshot(stage),
            ).progress
        )
    return max(0.0, min(1.0, progress / total))


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


def _python_rerun_line(snapshot: RunSnapshot, artifacts: dict[str, Any]) -> str | None:
    if artifacts.get("python_force_rerun_from"):
        return f"  Python rerun from: {artifacts.get('python_force_rerun_from')}"
    python_pipeline = snapshot.optional_tasks.get("python_pipeline")
    if python_pipeline is None:
        return None
    python_force_rerun_from = python_pipeline.artifacts.get("force_rerun_from", "")
    return f"  Python rerun from: {python_force_rerun_from}" if python_force_rerun_from else None


def _extend_matlab_resume_lines(lines: list[str], snapshot: RunSnapshot) -> None:
    matlab_status_task = snapshot.optional_tasks.get("matlab_status")
    if matlab_status_task is None:
        return
    artifacts = matlab_status_task.artifacts
    lines.extend(("", "MATLAB resume:"))
    lines.append(f"  Batch folder: {artifacts.get('batch_folder') or '(none)'}")
    lines.append(
        "  Resume mode: "
        f"{artifacts.get('resume_mode', 'unknown')}"
        f" | last completed={artifacts.get('last_completed_stage', '(none)')}"
        f" | next={artifacts.get('next_stage', '(none)')}"
    )
    python_rerun_line = _python_rerun_line(snapshot, artifacts)
    if python_rerun_line is not None:
        lines.append(python_rerun_line)
    if artifacts.get("rerun_prediction"):
        lines.append(f"  Prediction: {artifacts.get('rerun_prediction')}")
    if artifacts.get("failure_summary_file"):
        lines.append(f"  Failure summary file: {artifacts.get('failure_summary_file')}")


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
    _extend_matlab_resume_lines(lines, snapshot)
    if snapshot.errors:
        lines.extend(("", "Errors:"))
        lines.extend(
            f"  - {error.get('stage', 'run')}: {error.get('message', '')}"
            for error in snapshot.errors[-5:]
        )
    return lines
