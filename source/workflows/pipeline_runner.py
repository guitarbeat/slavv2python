"""Small orchestration helpers for sequential pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pipeline_results import stop_after_stage_if_requested
from .pipeline_setup import emit_progress

if TYPE_CHECKING:
    from collections.abc import Callable

    from slavv.runtime import RunContext


@dataclass(frozen=True)
class PipelineStageStep:
    """Description of one sequential stage in the pipeline runner."""

    result_key: str
    stage_name: str
    progress_fraction: float
    resolve_fn: Callable[[], dict[str, Any]]


def build_standard_pipeline_steps(
    *,
    resolve_energy: Callable[[], dict[str, Any]],
    resolve_vertices: Callable[[], dict[str, Any]],
    resolve_edges: Callable[[], dict[str, Any]],
    resolve_network: Callable[[], dict[str, Any]],
) -> list[PipelineStageStep]:
    """Return the canonical SLAVV stage sequence with shared metadata."""
    return [
        PipelineStageStep(
            result_key="energy_data",
            stage_name="energy",
            progress_fraction=0.4,
            resolve_fn=resolve_energy,
        ),
        PipelineStageStep(
            result_key="vertices",
            stage_name="vertices",
            progress_fraction=0.6,
            resolve_fn=resolve_vertices,
        ),
        PipelineStageStep(
            result_key="edges",
            stage_name="edges",
            progress_fraction=0.8,
            resolve_fn=resolve_edges,
        ),
        PipelineStageStep(
            result_key="network",
            stage_name="network",
            progress_fraction=1.0,
            resolve_fn=resolve_network,
        ),
    ]


def advance_pipeline_stage(
    results: dict[str, Any],
    *,
    result_key: str,
    payload: dict[str, Any],
    stage_name: str,
    progress_fraction: float,
    progress_callback: Callable[[float, str], None] | None,
    stop_after: str | None,
    run_context: RunContext | None,
) -> dict[str, Any] | None:
    """Record a stage payload, emit progress, and stop early if requested."""
    results[result_key] = payload
    emit_progress(progress_callback, progress_fraction, stage_name)
    return stop_after_stage_if_requested(stop_after, stage_name, results, run_context)


def run_pipeline_stage_sequence(
    results: dict[str, Any],
    *,
    steps: list[PipelineStageStep],
    progress_callback: Callable[[float, str], None] | None,
    stop_after: str | None,
    run_context: RunContext | None,
) -> dict[str, Any] | None:
    """Execute sequential pipeline steps until completion or an early stop."""
    for step in steps:
        payload = step.resolve_fn()
        if early_result := advance_pipeline_stage(
            results,
            result_key=step.result_key,
            payload=payload,
            stage_name=step.stage_name,
            progress_fraction=step.progress_fraction,
            progress_callback=progress_callback,
            stop_after=stop_after,
            run_context=run_context,
        ):
            return early_result
    return None


__all__ = [
    "PipelineStageStep",
    "advance_pipeline_stage",
    "build_standard_pipeline_steps",
    "run_pipeline_stage_sequence",
]
