"""Compatibility facade for flat pipeline setup helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from source import utils
from source.runtime.run_state import atomic_write_json, fingerprint_array, fingerprint_jsonable
from source.workflows.pipeline import session as _session
from source.workflows.pipeline.session import PreparedPipelineRun

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from source.runtime import ProgressEvent, RunContext


def validate_stage_control(stage_name: str | None, option_name: str) -> None:
    """Validate stop-after and force-rerun stage selectors."""
    _session.validate_stage_control(stage_name, option_name)


def emit_progress(
        progress_callback: Callable[[float, str], None] | None,
        fraction: float,
        stage: str,
) -> None:
    """Emit a pipeline progress update when a callback is present."""
    _session.emit_progress(progress_callback, fraction, stage)


def effective_run_dir(
        run_dir: str | None,
        event_callback: Callable[[ProgressEvent], None] | None,
) -> str | None:
    """Return the effective run directory for this pipeline execution."""
    return _session.effective_run_dir(run_dir, event_callback)


def create_run_context(
        effective_run_dir_value: str | None,
        input_fingerprint: str,
        params_fingerprint: str,
        image: np.ndarray,
        stop_after: str | None,
        event_callback: Callable[[ProgressEvent], None] | None,
        *,
        run_context_factory: type[RunContext] | Callable[..., RunContext],
) -> RunContext | None:
    """Create a run context when structured run tracking is enabled."""
    return _session.create_run_context(
        effective_run_dir_value,
        input_fingerprint,
        params_fingerprint,
        image,
        stop_after,
        event_callback,
        run_context_factory=run_context_factory,
    )


def initialize_run_context(
        run_context: RunContext | None,
        *,
        input_fingerprint: str,
        params_fingerprint: str,
        force_rerun_from: str | None,
        parameters: dict[str, Any],
) -> None:
    """Initialize the run context metadata and resume policy."""
    _session.initialize_run_context(
        run_context,
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
        force_rerun_from=force_rerun_from,
        parameters=parameters,
        atomic_write_json_fn=atomic_write_json,
    )


def force_rerun_flags(force_rerun_from: str | None) -> dict[str, bool]:
    """Return a per-stage map indicating which stages should be recomputed."""
    return _session.force_rerun_flags(force_rerun_from)


def preprocess_image(
        image: np.ndarray,
        parameters: dict[str, Any],
        run_context: RunContext | None,
) -> np.ndarray:
    """Preprocess the image and update run-state bookkeeping."""
    return _session.preprocess_image(
        image,
        parameters,
        run_context,
        preprocess_image_fn=utils.preprocess_image,
    )


def prepare_pipeline_run(
        image: np.ndarray,
        parameters: dict[str, Any],
        *,
        run_dir: str | None,
        stop_after: str | None,
        force_rerun_from: str | None,
        event_callback: Callable[[ProgressEvent], None] | None,
        run_context_factory: type[RunContext] | Callable[..., RunContext],
) -> PreparedPipelineRun:
    """Validate inputs and prepare run-state bookkeeping for pipeline execution."""
    return _session.prepare_pipeline_run(
        image,
        parameters,
        run_dir=run_dir,
        stop_after=stop_after,
        force_rerun_from=force_rerun_from,
        event_callback=event_callback,
        run_context_factory=run_context_factory,
        validate_parameters_fn=utils.validate_parameters,
        fingerprint_array_fn=fingerprint_array,
        fingerprint_jsonable_fn=fingerprint_jsonable,
        effective_run_dir_fn=effective_run_dir,
        create_run_context_fn=create_run_context,
        initialize_run_context_fn=initialize_run_context,
        force_rerun_flags_fn=force_rerun_flags,
    )


__all__ = [
    "PreparedPipelineRun",
    "create_run_context",
    "effective_run_dir",
    "emit_progress",
    "force_rerun_flags",
    "initialize_run_context",
    "prepare_pipeline_run",
    "preprocess_image",
    "validate_stage_control",
]
