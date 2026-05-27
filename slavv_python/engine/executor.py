"""Centralized pipeline stage execution engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from slavv_python.workflows.session import emit_progress

if TYPE_CHECKING:
    from collections.abc import Callable

    from slavv_python.engine.context import RunContext
    from slavv_python.engine.state import StageController
    from slavv_python.engine.state.run_state import RunState

logger = logging.getLogger(__name__)


class StageExecutor:
    """Consolidated engine for running resumable pipeline stages with full lifecycle management."""

    def __init__(
        self,
        run_context: RunContext | None,
        progress_callback: Callable[[float, str], None] | None = None,
        run_state: RunState | None = None,
    ):
        self.run_context = run_context
        self.progress_callback = progress_callback
        self.run_state = run_state

    def execute(
        self,
        stage_name: str,
        result_key: str,
        progress_fraction: float,
        compute_fn: Callable[[StageController], Any],
        fallback_fn: Callable[[], Any] | None = None,
        force_rerun: bool = False,
        log_label: str | None = None,
        schema_class: type[Any] | None = None,
    ) -> Any:
        """
        Execute a pipeline stage with checkpointing, logging, and progress tracking.

        Args:
            stage_name: Internal name of the stage (e.g. 'energy', 'vertices')
            result_key: Key in the results dict to store the payload
            progress_fraction: Target progress fraction (0.0 to 1.0)
            compute_fn: Resumable function receiving a StageController
            fallback_fn: Optional non-resumable fallback function
            force_rerun: If True, ignore existing checkpoints
            log_label: Friendly label for logging
            schema_class: Optional typed schema class to wrap the result
        """
        label = log_label or stage_name.capitalize()

        # 1. Handle No Context (Ephemeral Run)
        if self.run_context is None:
            logger.info("Running ephemeral stage: %s", label)
            payload = fallback_fn() if fallback_fn else compute_fn(None)  # type: ignore
            if (
                schema_class
                and not isinstance(payload, schema_class)
                and hasattr(schema_class, "from_dict")
            ):
                payload = schema_class.from_dict(payload)
            return self._finalize_step(result_key, payload, stage_name, progress_fraction)

        # 2. Resumable Logic
        controller = self.run_context.stage(stage_name)
        try:
            # Check for existing checkpoint
            if controller.checkpoint_path.exists() and not force_rerun:
                logger.info("Loading cached %s from %s", label, controller.checkpoint_path)
                payload = self._load_checkpoint(controller, schema_class)

                controller.complete(
                    detail=f"Loaded cached {label}",
                    artifacts=self._get_artifacts(controller),
                    resumed=True,
                )
            else:
                # Compute fresh
                logger.info("Computing stage: %s", label)
                payload = compute_fn(controller)

                self._save_checkpoint(controller, payload)
                controller.complete(
                    detail=f"{label} ready",
                    artifacts=self._get_artifacts(controller),
                )

            return self._finalize_step(result_key, payload, stage_name, progress_fraction)

        except Exception as exc:
            logger.error("Stage '%s' failed: %s", stage_name, exc)
            self.run_context.fail_stage(stage_name, exc)
            raise

    def _finalize_step(
        self, result_key: str, payload: Any, stage_name: str, progress_fraction: float
    ) -> Any:
        """Record result, emit progress, and return payload."""
        if self.run_state is not None:
            self.run_state.set_result(result_key, payload)
        emit_progress(self.progress_callback, progress_fraction, stage_name)
        return payload

    def _load_checkpoint(self, controller: StageController, schema_class: type[Any] | None) -> Any:
        if schema_class is not None and hasattr(schema_class, "load"):
            return schema_class.load(controller.checkpoint_path)
        payload = controller.load_checkpoint()
        if (
            schema_class is not None
            and not isinstance(payload, schema_class)
            and hasattr(schema_class, "from_dict")
        ):
            return schema_class.from_dict(payload)
        return payload

    def _save_checkpoint(self, controller: StageController, payload: Any) -> None:
        if hasattr(payload, "save"):
            payload.save(controller.checkpoint_path)
            return
        if hasattr(payload, "to_dict"):
            controller.save_checkpoint(payload.to_dict())
            return
        controller.save_checkpoint(payload)

    def _get_artifacts(self, controller: StageController) -> dict[str, str]:
        """Collect artifacts from stage directory."""
        from pathlib import Path

        artifacts: dict[str, str] = {}
        stage_dir = Path(controller.stage_dir)
        if stage_dir.exists():
            for artifact in stage_dir.iterdir():
                if artifact.name == "resume_state.json":
                    continue
                if artifact.is_file() or artifact.is_dir():
                    artifacts[artifact.name] = str(artifact)
        return artifacts
