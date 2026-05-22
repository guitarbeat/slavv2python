"""Compatibility facade for flat pipeline setup helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python import utils
from slavv_python.workflows.pipeline import session as _session
from slavv_python.workflows.pipeline.session import PreparedPipelineRun

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from slavv_python.engine.state import ProgressEvent, RunContext


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
    from slavv_python.engine.context import RunContext as ActualRunContext

    # We ignore run_context_factory here and use the actual RunContext.prepare
    # since it's the authoritative deep interface now.
    validated_params, context, rerun_flags = ActualRunContext.prepare(
        image,
        parameters,
        run_dir=run_dir,
        stop_after=stop_after,
        force_rerun_from=force_rerun_from,
        event_callback=event_callback,
    )
    return PreparedPipelineRun(
        parameters=validated_params,
        run_context=context,
        force_rerun=rerun_flags,
    )


__all__ = [
    "PreparedPipelineRun",
    "emit_progress",
    "prepare_pipeline_run",
    "preprocess_image",
    "validate_stage_control",
]
