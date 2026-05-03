"""Shared helpers for pipeline stage resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .artifacts import resolve_resumable_stage

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
        resolve_resumable_stage_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a pipeline stage with fallback, checkpoint reuse, and failure tracking."""
    if resolve_resumable_stage_fn is None:
        resolve_resumable_stage_fn = resolve_resumable_stage

    if run_context is None:
        return fallback_fn()

    controller = run_context.stage(stage_name)
    try:
        return resolve_resumable_stage_fn(
            controller,
            force_rerun=force_rerun,
            cached_log_label=cached_log_label,
            cached_detail=cached_detail,
            success_detail=success_detail,
            compute_fn=lambda: compute_fn(controller),
            logger=logger,
        )
    except Exception as exc:
        run_context.fail_stage(stage_name, exc)
        raise


def resolve_energy_stage(
        *,
        run_context,
        force_rerun: bool,
        fallback_fn: Callable[[], dict[str, Any]],
        resumable_fn: Callable[[Any], dict[str, Any]],
        logger,
        resolve_stage_with_checkpoint_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the energy stage using the standard checkpoint contract."""
    if resolve_stage_with_checkpoint_fn is None:
        resolve_stage_with_checkpoint_fn = resolve_stage_with_checkpoint

    return resolve_stage_with_checkpoint_fn(
        run_context=run_context,
        force_rerun=force_rerun,
        stage_name="energy",
        cached_log_label="Energy Field",
        cached_detail="Loaded energy checkpoint",
        success_detail="Energy field ready",
        fallback_fn=fallback_fn,
        compute_fn=resumable_fn,
        logger=logger,
    )


def resolve_vertices_stage(
        *,
        run_context,
        force_rerun: bool,
        fallback_fn: Callable[[], dict[str, Any]],
        resumable_fn: Callable[[Any], dict[str, Any]],
        logger,
        resolve_stage_with_checkpoint_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the vertices stage using the standard checkpoint contract."""
    if resolve_stage_with_checkpoint_fn is None:
        resolve_stage_with_checkpoint_fn = resolve_stage_with_checkpoint

    return resolve_stage_with_checkpoint_fn(
        run_context=run_context,
        force_rerun=force_rerun,
        stage_name="vertices",
        cached_log_label="Vertices",
        cached_detail="Loaded vertex checkpoint",
        success_detail="Vertices extracted",
        fallback_fn=fallback_fn,
        compute_fn=resumable_fn,
        logger=logger,
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
        resolve_stage_with_checkpoint_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the edges stage while preserving the tracing/watershed switch."""
    if resolve_stage_with_checkpoint_fn is None:
        resolve_stage_with_checkpoint_fn = resolve_stage_with_checkpoint

    use_watershed = edge_method == "watershed"
    return resolve_stage_with_checkpoint_fn(
        run_context=run_context,
        force_rerun=force_rerun,
        stage_name="edges",
        cached_log_label="Edges",
        cached_detail="Loaded edge checkpoint",
        success_detail="Edges extracted",
        fallback_fn=watershed_fallback_fn if use_watershed else tracing_fallback_fn,
        compute_fn=watershed_resumable_fn if use_watershed else tracing_resumable_fn,
        logger=logger,
    )


def resolve_network_stage(
        *,
        run_context,
        force_rerun: bool,
        fallback_fn: Callable[[], dict[str, Any]],
        resumable_fn: Callable[[Any], dict[str, Any]],
        logger,
        resolve_stage_with_checkpoint_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the network stage using the standard checkpoint contract."""
    if resolve_stage_with_checkpoint_fn is None:
        resolve_stage_with_checkpoint_fn = resolve_stage_with_checkpoint

    return resolve_stage_with_checkpoint_fn(
        run_context=run_context,
        force_rerun=force_rerun,
        stage_name="network",
        cached_log_label="Network",
        cached_detail="Loaded network checkpoint",
        success_detail="Network constructed",
        fallback_fn=fallback_fn,
        compute_fn=resumable_fn,
        logger=logger,
    )


__all__ = [
    "resolve_edges_stage",
    "resolve_energy_stage",
    "resolve_network_stage",
    "resolve_stage_with_checkpoint",
    "resolve_vertices_stage",
]
