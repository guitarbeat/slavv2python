"""File-backed run state for resumable SLAVV processing."""

from __future__ import annotations

import calendar
import copy
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, cast

from slavv.utils.safe_unpickle import safe_load

from ._run_state.constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STAGE_WEIGHTS,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
)
from ._run_state.io import (
    _ensure_stage_map,
    _normalize_for_json,
    atomic_joblib_dump,
    atomic_write_json,
    load_legacy_run_snapshot,
    load_run_snapshot,
)
from ._run_state.io import (
    fingerprint_array as _fingerprint_array,
)
from ._run_state.io import (
    fingerprint_file as _fingerprint_file,
)
from ._run_state.io import (
    fingerprint_jsonable as _fingerprint_jsonable,
)
from ._run_state.io import (
    stable_json_dumps as _stable_json_dumps,
)
from ._run_state.models import ProgressEvent, RunSnapshot, StageSnapshot, TaskSnapshot, _now_iso

logger = logging.getLogger(__name__)

fingerprint_array = _fingerprint_array
fingerprint_file = _fingerprint_file
fingerprint_jsonable = _fingerprint_jsonable
stable_json_dumps = _stable_json_dumps


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
        return float(remaining)

    def _estimate_run_eta(self, snapshot: RunSnapshot) -> float | None:
        if snapshot.overall_progress <= 0.0:
            return None
        created = self._parse_time(snapshot.created_at)
        if created is None:
            return None
        elapsed = max(0.0, time.time() - created)
        snapshot.elapsed_seconds = elapsed
        return float(elapsed * (1.0 - snapshot.overall_progress) / snapshot.overall_progress)

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
        return float(max(0.0, min(1.0, progress / total)))

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
        return float(snapshot.overall_progress)
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
    return float(max(0.0, min(1.0, progress / total)))


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
