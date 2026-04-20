"""Shared helpers for pipeline stage resolution."""

from __future__ import annotations

from typing import Any

from .stage_checkpoints import resolve_resumable_stage


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
    if run_context is None:
        return fallback_fn()

    controller = run_context.stage(stage_name)
    try:
        return resolve_resumable_stage(
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


__all__ = ["resolve_stage_with_checkpoint"]
