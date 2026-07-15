"""Setup helpers for pipeline orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from slavv_python import utils
from slavv_python.engine.constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from slavv_python.engine.state import RunContext


def validate_stage_control(stage_name: str | None, option_name: str) -> None:
    """Validate stop-after and force-rerun stage selectors."""
    allowed = PIPELINE_STAGES if option_name == "force_rerun_from" else [*PIPELINE_STAGES, PREPROCESS_STAGE]
    if stage_name is not None and stage_name not in allowed:
        valid = ", ".join(allowed)
        raise ValueError(f"{option_name} must be one of: {valid}")


def emit_progress(
    progress_callback: Callable[[float, str], None] | None,
    fraction: float,
    stage: str,
) -> None:
    """Emit a pipeline progress update when a callback is present."""
    if progress_callback:
        progress_callback(fraction, stage)


def preprocess_image(
    image: np.ndarray,
    parameters: dict[str, Any],
    run_context: RunContext | None,
    *,
    preprocess_image_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None,
) -> np.ndarray:
    """Preprocess the image and update run-state bookkeeping."""
    if preprocess_image_fn is None:
        preprocess_image_fn = utils.preprocess_image

    try:
        preprocessed = cast("np.ndarray", preprocess_image_fn(image, parameters))
    except Exception as exc:
        if run_context is not None:
            run_context.fail_stage(PREPROCESS_STAGE, exc)
        raise
    if run_context is not None:
        run_context.mark_preprocess_complete()
    return preprocessed


__all__ = [
    "emit_progress",
    "preprocess_image",
    "validate_stage_control",
]
