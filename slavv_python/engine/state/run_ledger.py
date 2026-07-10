from __future__ import annotations

import logging
import subprocess
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import psutil

from slavv_python.engine.constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from slavv_python.engine.state.io import (
    atomic_write_json,
    fingerprint_array,
    fingerprint_jsonable,
    load_json_dict,
)
from slavv_python.engine.state.layout import resolve_run_layout
from slavv_python.engine.state.models import (  # noqa: TC001
    ProgressEvent,
    RunSnapshot,
    StageSnapshot,
)
from slavv_python.engine.state.progress import (
    calculate_overall_progress,
    estimate_run_eta,
    estimate_stage_eta,
    parse_run_time,
    preprocess_complete,
)
from slavv_python.engine.state.reset import clear_stage_runtime_artifacts, reset_stage_snapshots
from slavv_python.engine.state.resume_policy import (
    apply_resume_block,
    apply_resume_reset,
    fingerprint_mismatches,
    update_snapshot_fingerprints,
)
from slavv_python.engine.state.snapshot_lifecycle import (
    begin_stage_snapshot,
    complete_stage_snapshot,
    fail_stage_snapshot,
    finalize_run_snapshot,
    mark_preprocess_complete_snapshot,
    update_optional_task_snapshot,
    update_stage_snapshot,
)
from slavv_python.engine.state.snapshots import (
    emit_progress_event,
    load_or_create_snapshot,
    persist_snapshot,
)
from slavv_python.engine.state.stage_handle import StageController

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_METRIC_STAGES: tuple[str, ...] = ("energy", "vertices", "edges", "network")


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
    def prepare(
        cls,
        image: np.ndarray,
        parameters: dict[str, Any],
        *,
        run_dir: str | Path | None = None,
        stop_after: str | None = None,
        force_rerun_from: str | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> tuple[dict[str, Any], RunContext | None, dict[str, bool]]:
        """
        Validate inputs and prepare the run context for pipeline execution.

        Returns:
            (validated_parameters, run_context, force_rerun_flags)
        """
        from slavv_python.utils import validate_parameters

        # 1. Resolve Run Directory
        effective_dir = run_dir
        if effective_dir is None and event_callback is not None:
            import tempfile

            effective_dir = tempfile.mkdtemp(prefix="slavv_run_")

        # 2. Validate and fingerprint the caller's parameters before resume adoption.
        validated_incoming = validate_parameters(parameters)
        input_hash = fingerprint_array(image)
        params_hash_incoming = fingerprint_jsonable(validated_incoming)

        existing_params: dict[str, Any] | None = None
        if effective_dir and not force_rerun_from:
            loaded = load_json_dict(Path(effective_dir) / "99_Metadata" / "validated_params.json")
            existing_params = loaded if loaded else None

        # 3. Create Context
        context = None
        if effective_dir:
            context = cls(
                run_dir=effective_dir,
                input_fingerprint=input_hash,
                params_fingerprint=params_hash_incoming,
                target_stage=stop_after or "network",
                provenance={
                    "slavv_python": "pipeline",
                    "image_shape": list(image.shape),
                    "stop_after": stop_after or "network",
                },
                event_callback=event_callback,
            )

            # 4. Resume policy uses incoming fingerprints so parameter changes are blocked.
            context.ensure_resume_allowed(
                input_fingerprint=input_hash,
                params_fingerprint=params_hash_incoming,
                force_rerun_from=force_rerun_from,
            )

            if existing_params:
                logger.info(
                    "Adopting existing parameters from %s for resume compatibility", effective_dir
                )
                validated_params = validate_parameters(existing_params)
            else:
                validated_params = validated_incoming
            if force_rerun_from is not None and force_rerun_from in PIPELINE_STAGES:
                context.reset_pipeline_state_from(force_rerun_from)

            params_path = context.metadata_dir / "validated_params.json"
            atomic_write_json(params_path, validated_params)
            context.mark_run_status(
                STATUS_RUNNING,
                current_stage=PREPROCESS_STAGE,
                detail="Starting SLAVV processing pipeline",
            )
        else:
            validated_params = validated_incoming

        # 5. Calculate Force Rerun Flags
        rerun_flags = dict.fromkeys(PIPELINE_STAGES, False)
        if force_rerun_from in PIPELINE_STAGES:
            start_idx = PIPELINE_STAGES.index(force_rerun_from)
            for stage_name in PIPELINE_STAGES[start_idx:]:
                rerun_flags[stage_name] = True

        return validated_params, context, rerun_flags

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
        stage_snapshot = self.snapshot.stages.get(stage)
        traceback_text = (
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
            if isinstance(error, BaseException)
            else None
        )
        fail_stage_snapshot(
            self.snapshot,
            stage=stage,
            message=message,
            substage=stage_snapshot.substage if stage_snapshot is not None else "",
            traceback_text=traceback_text,
        )
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
        if stop_after is not None and stop_after in PIPELINE_STAGES:
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
