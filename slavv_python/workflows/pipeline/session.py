"""Setup helpers for pipeline orchestration."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from slavv_python import utils
from slavv_python.runtime.run_state import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STATUS_RUNNING,
    atomic_write_json,
    fingerprint_array,
    fingerprint_jsonable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from slavv_python.runtime import ProgressEvent, RunContext


@dataclass(frozen=True)
class PreparedPipelineRun:
    """Prepared pipeline execution state before stage resolution begins."""

    parameters: dict[str, Any]
    run_context: RunContext | None
    force_rerun: dict[str, bool]


def validate_stage_control(stage_name: str | None, option_name: str) -> None:
    """Validate stop-after and force-rerun stage selectors."""
    if stage_name is not None and stage_name not in PIPELINE_STAGES:
        valid = ", ".join(PIPELINE_STAGES)
        raise ValueError(f"{option_name} must be one of: {valid}")


def emit_progress(
    progress_callback: Callable[[float, str], None] | None,
    fraction: float,
    stage: str,
) -> None:
    """Emit a pipeline progress update when a callback is present."""
    if progress_callback:
        progress_callback(fraction, stage)


def effective_run_dir(
    run_dir: str | None,
    event_callback: Callable[[ProgressEvent], None] | None,
) -> str | None:
    """Return the effective run directory for this pipeline execution."""
    if run_dir is not None or event_callback is None:
        return run_dir
    return tempfile.mkdtemp(prefix="slavv_run_")


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
    if effective_run_dir_value is None:
        return None
    return run_context_factory(
        run_dir=effective_run_dir_value,
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
        target_stage=stop_after or "network",
        provenance={
            "source": "pipeline",
            "image_shape": list(image.shape),
            "stop_after": stop_after or "network",
        },
        event_callback=event_callback,
    )


def initialize_run_context(
    run_context: RunContext | None,
    *,
    input_fingerprint: str,
    params_fingerprint: str,
    force_rerun_from: str | None,
    parameters: dict[str, Any],
    atomic_write_json_fn: Callable[[Any, Any], None] | None = None,
) -> None:
    """Initialize the run context metadata and resume policy."""
    if atomic_write_json_fn is None:
        atomic_write_json_fn = atomic_write_json

    if run_context is None:
        return
    run_context.ensure_resume_allowed(
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
        force_rerun_from=force_rerun_from,
    )
    if force_rerun_from in PIPELINE_STAGES:
        run_context.reset_pipeline_state_from(force_rerun_from)
    params_path = run_context.metadata_dir / "validated_params.json"
    atomic_write_json_fn(params_path, parameters)
    run_context.mark_run_status(
        STATUS_RUNNING,
        current_stage=PREPROCESS_STAGE,
        detail="Starting SLAVV processing pipeline",
    )


def force_rerun_flags(force_rerun_from: str | None) -> dict[str, bool]:
    """Return a per-stage map indicating which stages should be recomputed."""
    rerun = dict.fromkeys(PIPELINE_STAGES, False)
    if force_rerun_from not in PIPELINE_STAGES:
        return rerun
    start_idx = PIPELINE_STAGES.index(force_rerun_from)
    for stage_name in PIPELINE_STAGES[start_idx:]:
        rerun[stage_name] = True
    return rerun


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


def prepare_pipeline_run(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    run_dir: str | None,
    stop_after: str | None,
    force_rerun_from: str | None,
    event_callback: Callable[[ProgressEvent], None] | None,
    run_context_factory: type[RunContext] | Callable[..., RunContext],
    validate_parameters_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    fingerprint_array_fn: Callable[[np.ndarray], str] | None = None,
    fingerprint_jsonable_fn: Callable[[dict[str, Any]], str] | None = None,
    effective_run_dir_fn: Callable[..., str | None] | None = None,
    create_run_context_fn: Callable[..., RunContext | None] | None = None,
    initialize_run_context_fn: Callable[..., None] | None = None,
    force_rerun_flags_fn: Callable[[str | None], dict[str, bool]] | None = None,
) -> PreparedPipelineRun:
    """Validate inputs and prepare run-state bookkeeping for pipeline execution."""
    if validate_parameters_fn is None:
        validate_parameters_fn = utils.validate_parameters
    if fingerprint_array_fn is None:
        fingerprint_array_fn = fingerprint_array
    if fingerprint_jsonable_fn is None:
        fingerprint_jsonable_fn = fingerprint_jsonable
    if effective_run_dir_fn is None:
        effective_run_dir_fn = effective_run_dir
    if create_run_context_fn is None:
        create_run_context_fn = create_run_context
    if initialize_run_context_fn is None:
        initialize_run_context_fn = initialize_run_context
    if force_rerun_flags_fn is None:
        force_rerun_flags_fn = force_rerun_flags

    validated_parameters = validate_parameters_fn(parameters)
    input_fingerprint = fingerprint_array_fn(image)
    params_fingerprint = fingerprint_jsonable_fn(validated_parameters)
    resolved_run_dir = effective_run_dir_fn(run_dir, event_callback)
    run_context = create_run_context_fn(
        resolved_run_dir,
        input_fingerprint,
        params_fingerprint,
        image,
        stop_after,
        event_callback,
        run_context_factory=run_context_factory,
    )
    initialize_run_context_fn(
        run_context,
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
        force_rerun_from=force_rerun_from,
        parameters=validated_parameters,
    )
    return PreparedPipelineRun(
        parameters=validated_parameters,
        run_context=run_context,
        force_rerun=force_rerun_flags_fn(force_rerun_from),
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
