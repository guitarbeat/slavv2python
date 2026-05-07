"""Compatibility facade for flat pipeline stage-resolution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.workflows.pipeline import resolution as _resolution
from slavv_python.workflows.pipeline.artifacts import resolve_resumable_stage

if TYPE_CHECKING:
    from collections.abc import Callable


def resolve_stage_with_checkpoint(
    *,
    run_context,
    force_rerun: bool,
    stage_name: str,
    cached_log_label: str,
    cached_detail: str,
    success_detail: str,
    fallback_fn,
    compute_fn,
    logger,
) -> dict[str, Any]:
    """Resolve a pipeline stage with fallback, checkpoint reuse, and failure tracking."""
    return _resolution.resolve_stage_with_checkpoint(
        run_context=run_context,
        force_rerun=force_rerun,
        stage_name=stage_name,
        cached_log_label=cached_log_label,
        cached_detail=cached_detail,
        success_detail=success_detail,
        fallback_fn=fallback_fn,
        compute_fn=compute_fn,
        logger=logger,
        resolve_resumable_stage_fn=resolve_resumable_stage,
    )


def resolve_energy_stage(
    *,
    run_context,
    force_rerun: bool,
    fallback_fn: Callable[[], dict[str, Any]],
    resumable_fn: Callable[[Any], dict[str, Any]],
    logger,
) -> dict[str, Any]:
    """Resolve the energy stage using the standard checkpoint contract."""
    return _resolution.resolve_energy_stage(
        run_context=run_context,
        force_rerun=force_rerun,
        fallback_fn=fallback_fn,
        resumable_fn=resumable_fn,
        logger=logger,
        resolve_stage_with_checkpoint_fn=resolve_stage_with_checkpoint,
    )


def resolve_vertices_stage(
    *,
    run_context,
    force_rerun: bool,
    fallback_fn: Callable[[], dict[str, Any]],
    resumable_fn: Callable[[Any], dict[str, Any]],
    logger,
) -> dict[str, Any]:
    """Resolve the vertices stage using the standard checkpoint contract."""
    return _resolution.resolve_vertices_stage(
        run_context=run_context,
        force_rerun=force_rerun,
        fallback_fn=fallback_fn,
        resumable_fn=resumable_fn,
        logger=logger,
        resolve_stage_with_checkpoint_fn=resolve_stage_with_checkpoint,
    )


def resolve_edges_stage(
    *,
    run_context,
    force_rerun: bool,
    edge_method: str,
    tracing_fallback_fn: Callable[[], dict[str, Any]],
    watershed_fallback_fn: Callable[[], dict[str, Any]],
    tracing_resumable_fn: Callable[[Any], dict[str, Any]],
    watershed_resumable_fn: Callable[[Any], dict[str, Any]],
    logger,
) -> dict[str, Any]:
    """Resolve the edges stage while preserving the tracing/watershed switch."""
    return _resolution.resolve_edges_stage(
        run_context=run_context,
        force_rerun=force_rerun,
        edge_method=edge_method,
        tracing_fallback_fn=tracing_fallback_fn,
        watershed_fallback_fn=watershed_fallback_fn,
        tracing_resumable_fn=tracing_resumable_fn,
        watershed_resumable_fn=watershed_resumable_fn,
        logger=logger,
        resolve_stage_with_checkpoint_fn=resolve_stage_with_checkpoint,
    )


def resolve_network_stage(
    *,
    run_context,
    force_rerun: bool,
    fallback_fn: Callable[[], dict[str, Any]],
    resumable_fn: Callable[[Any], dict[str, Any]],
    logger,
) -> dict[str, Any]:
    """Resolve the network stage using the standard checkpoint contract."""
    return _resolution.resolve_network_stage(
        run_context=run_context,
        force_rerun=force_rerun,
        fallback_fn=fallback_fn,
        resumable_fn=resumable_fn,
        logger=logger,
        resolve_stage_with_checkpoint_fn=resolve_stage_with_checkpoint,
    )


__all__ = [
    "resolve_edges_stage",
    "resolve_energy_stage",
    "resolve_network_stage",
    "resolve_stage_with_checkpoint",
    "resolve_vertices_stage",
]
