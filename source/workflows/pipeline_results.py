"""Result normalization helpers for pipeline orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from slavv.models import normalize_pipeline_result

if TYPE_CHECKING:
    from slavv.runtime import RunContext

logger = logging.getLogger(__name__)


def finalize_pipeline_results(results: dict[str, Any]) -> dict[str, Any]:
    """Normalize pipeline results through the typed compatibility adapter."""
    return normalize_pipeline_result(results).to_dict()


def stop_after_stage_if_requested(
    stop_after: str | None,
    stage_name: str,
    results: dict[str, Any],
    run_context: RunContext | None,
) -> dict[str, Any] | None:
    """Finalize and return results when the requested stop stage has completed."""
    if stop_after != stage_name:
        return None

    logger.info("Pipeline stopped after '%s' stage as requested.", stage_name)
    if run_context is not None:
        run_context.finalize_run(stop_after=stop_after)
    return finalize_pipeline_results(results)
