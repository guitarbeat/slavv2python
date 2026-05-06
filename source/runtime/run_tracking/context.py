from __future__ import annotations

import json
import logging
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast

import psutil

from source.utils.safe_unpickle import safe_load

from .constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from .io import (
    atomic_joblib_dump,
    atomic_write_json,
)
from .layout import resolve_run_layout
from .lifecycle import (
    begin_stage_snapshot,
    complete_stage_snapshot,
    fail_stage_snapshot,
    finalize_run_snapshot,
    mark_preprocess_complete_snapshot,
    update_optional_task_snapshot,
    update_stage_snapshot,
)
from .models import ProgressEvent, RunSnapshot, StageSnapshot, _now_iso
from .progress import (
    calculate_overall_progress,
    estimate_run_eta,
    estimate_stage_eta,
    parse_run_time,
    preprocess_complete,
)
from .reset import clear_stage_runtime_artifacts, reset_stage_snapshots
from .resume_guard import (
    apply_resume_block,
    apply_resume_reset,
    fingerprint_mismatches,
    update_snapshot_fingerprints,
)
from .snapshot_store import emit_progress_event, load_or_create_snapshot, persist_snapshot

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_METRIC_STAGES: tuple[str, ...] = ("energy", "vertices", "edges", "network")


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


class RunContext:
    """Shared run ledger and file layout for resumable processing."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        input_fingerprint: str = "",
        params_fingerprint: str = "",
        target_stage: str | None = "network",
        provenance: dict[str, Any] | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
    ):
        self.layout = resolve_run_layout(
            run_dir=run_dir,
        )
        self.run_root = self.layout.run_root
        self.refs_dir = self.layout.refs_dir
        self.params_dir = self.layout.params_dir
        self.metadata_dir = self.layout.metadata_dir
        self.artifacts_dir = self.layout.artifacts_dir
        self.analysis_dir = self.layout.analysis_dir
        self.stages_dir = self.layout.stages_dir
        self.checkpoints_dir = self.layout.checkpoints_dir
        self.snapshot_path = self.layout.snapshot_path
        self.manifest_path = self.layout.manifest_path
        self.event_callback = event_callback
        self.start_time = time.time()
        self.snapshot = load_or_create_snapshot(
            self.layout,
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
        target_stage: str | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> RunContext:
        return cls(
            run_dir=run_dir,
            target_stage=target_stage,
            event_callback=event_callback,
        )

    def checkpoint_path(self, stage: str) -> Path:
        return self.layout.checkpoint_path(stage)

    def stage(self, name: str) -> StageController:
        if name not in PIPELINE_STAGES:
            valid = ", ".join(PIPELINE_STAGES)
            raise ValueError(f"stage must be one of: {valid}")
        return StageController(self, name)

    def persist(self) -> None:
        persist_snapshot(self.snapshot, self.snapshot_path, start_time=self.start_time)
        atomic_write_json(self.manifest_path, self._build_run_manifest())

    def ensure_resume_allowed(
        self,
        *,
        input_fingerprint: str,
        params_fingerprint: str,
        force_rerun_from: str | None = None,
    ) -> None:
        mismatch = fingerprint_mismatches(
            self.snapshot,
            input_fingerprint=input_fingerprint,
            params_fingerprint=params_fingerprint,
        )

        if not mismatch:
            update_snapshot_fingerprints(
                self.snapshot,
                input_fingerprint=input_fingerprint,
                params_fingerprint=params_fingerprint,
            )
            self.persist()
            return

        if force_rerun_from == "energy":
            apply_resume_reset(
                self.snapshot,
                input_fingerprint=input_fingerprint,
                params_fingerprint=params_fingerprint,
                mismatch=mismatch,
                logger=logger,
            )
            self.reset_pipeline_state()
            self.persist()
            return

        message = apply_resume_block(self.snapshot, mismatch)
        self.persist()
        raise RuntimeError(message)

    def reset_pipeline_state(self) -> None:
        self.reset_pipeline_state_from("energy")

    def reset_pipeline_state_from(self, start_stage: str) -> None:
        affected_stages = reset_stage_snapshots(self.snapshot.stages, start_stage=start_stage)
        if not affected_stages:
            return
        for stage in affected_stages:
            controller = self.stage(stage)
            clear_stage_runtime_artifacts(controller)
        self.persist()

    def mark_preprocess_complete(self) -> None:
        mark_preprocess_complete_snapshot(
            self.snapshot,
            overall_progress=self._calculate_overall_progress(
                self.snapshot.stages, preprocess_done=True
            ),
        )
        self._refresh_stage_metrics(PREPROCESS_STAGE)
        self.persist()
        self.emit_event(PREPROCESS_STAGE, STATUS_COMPLETED, detail="Preprocessing complete")

    def mark_run_status(self, status: str, *, current_stage: str = "", detail: str = "") -> None:
        self.snapshot.status = status
        if current_stage:
            self.snapshot.current_stage = current_stage
        if detail:
            self.snapshot.current_detail = detail
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
        begin_stage_snapshot(
            self.snapshot,
            stage=stage,
            detail=detail,
            units_total=units_total,
            units_completed=units_completed,
            substage=substage,
            resumed=resumed,
        )
        self._refresh_stage_metrics(stage)
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
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
        stage_snapshot = update_stage_snapshot(
            self.snapshot,
            stage=stage,
            detail=detail,
            units_total=units_total,
            units_completed=units_completed,
            progress=progress,
            substage=substage,
            resumed=resumed,
        )
        stage_snapshot.eta_seconds = self._estimate_eta(stage_snapshot)
        self._refresh_stage_metrics(stage)
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
        self.snapshot.eta_seconds = self._estimate_run_eta(self.snapshot)
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
        stage_snapshot = complete_stage_snapshot(
            self.snapshot,
            stage=stage,
            detail=detail,
            artifacts=artifacts,
            resumed=resumed,
        )
        self._refresh_stage_metrics(stage)
        self.snapshot.overall_progress = self._calculate_overall_progress(self.snapshot.stages)
        self.snapshot.eta_seconds = self._estimate_run_eta(self.snapshot)
        self.persist()
        self.emit_event(stage, STATUS_COMPLETED, detail=stage_snapshot.detail)

    def fail_stage(self, stage: str, error: BaseException | str) -> None:
        message = str(error)
        fail_stage_snapshot(self.snapshot, stage=stage, message=message)
        self._refresh_stage_metrics(stage)
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
        update_optional_task_snapshot(
            self.snapshot,
            name=name,
            status=status,
            detail=detail,
            progress=progress,
            artifacts=artifacts,
        )
        self.persist()

    def finalize_run(self, *, stop_after: str | None = None) -> None:
        if stop_after in PIPELINE_STAGES:
            self._refresh_stage_metrics(stop_after)
        elif self.snapshot.current_stage in PIPELINE_STAGES:
            self._refresh_stage_metrics(self.snapshot.current_stage)
        finalize_run_snapshot(self.snapshot, stop_after=stop_after)
        self.persist()
        self.emit_event(self.snapshot.current_stage, self.snapshot.status, detail="Run completed")

    def emit_event(self, stage: str, status: str, *, detail: str = "") -> None:
        emit_progress_event(
            self.snapshot,
            self.event_callback,
            stage=stage,
            status=status,
            detail=detail,
        )

    def _estimate_eta(self, stage_snapshot: StageSnapshot) -> float | None:
        return estimate_stage_eta(stage_snapshot)

    def _estimate_run_eta(self, snapshot: RunSnapshot) -> float | None:
        return estimate_run_eta(snapshot)

    def _calculate_overall_progress(
        self,
        stages: dict[str, StageSnapshot],
        preprocess_done: bool | None = None,
    ) -> float:
        if preprocess_done is None:
            preprocess_done = preprocess_complete(stages, snapshot=getattr(self, "snapshot", None))
        return calculate_overall_progress(stages, preprocess_done=preprocess_done)

    @staticmethod
    def _parse_time(timestamp: str | None) -> float | None:
        return parse_run_time(timestamp)

    def _refresh_stage_metrics(self, stage: str) -> None:
        stage_snapshot = self.snapshot.stages.get(stage)
        if stage_snapshot is None:
            return
        stage_snapshot.peak_memory_bytes = max(
            int(stage_snapshot.peak_memory_bytes),
            int(self._sample_process_memory_bytes()),
        )
        started = self._parse_time(stage_snapshot.started_at)
        if started is None:
            return
        finished = self._parse_time(stage_snapshot.completed_at)
        if finished is None:
            finished = time.time()
        stage_snapshot.elapsed_seconds = max(
            float(stage_snapshot.elapsed_seconds),
            float(max(0.0, finished - started)),
        )

    def _build_run_manifest(self) -> dict[str, Any]:
        provenance = cast("dict[str, Any]", dict(self.snapshot.provenance))
        return {
            "manifest_version": 1,
            "kind": "slavv_run",
            "run_id": self.snapshot.run_id,
            "run_root": str(self.run_root),
            "status": self.snapshot.status,
            "target_stage": self.snapshot.target_stage,
            "current_stage": self.snapshot.current_stage,
            "dataset_hash": provenance.get("dataset_hash")
            or self.snapshot.input_fingerprint
            or None,
            "params_fingerprint": self.snapshot.params_fingerprint or None,
            "oracle_id": provenance.get("oracle_id"),
            "python_commit": provenance.get("python_commit") or self._resolve_python_commit(),
            "matlab_source_version": provenance.get("matlab_source_version"),
            "retention": provenance.get("retention", "disposable"),
            "promotion_state": provenance.get("promotion_state", "ephemeral"),
            "timestamps": {
                "created_at": self.snapshot.created_at,
                "updated_at": self.snapshot.updated_at,
                "completed_at": max(
                    (
                        stage_snapshot.completed_at
                        for stage_snapshot in self.snapshot.stages.values()
                        if stage_snapshot.completed_at
                    ),
                    default=None,
                ),
            },
            "stage_metrics": {
                stage_name: {
                    "status": stage_snapshot.status,
                    "elapsed_seconds": float(stage_snapshot.elapsed_seconds),
                    "peak_memory_bytes": int(stage_snapshot.peak_memory_bytes),
                    "completed_at": stage_snapshot.completed_at,
                }
                for stage_name, stage_snapshot in self.snapshot.stages.items()
                if stage_name in MANIFEST_METRIC_STAGES
            },
            "provenance": provenance,
        }

    @staticmethod
    def _sample_process_memory_bytes() -> int:
        try:
            return int(psutil.Process().memory_info().rss)
        except (psutil.Error, OSError):
            return 0

    @staticmethod
    @lru_cache(maxsize=1)
    def _resolve_python_commit() -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                check=False,
                encoding="utf-8",
            )
        except OSError:
            return None
        commit = completed.stdout.strip()
        return commit or None
