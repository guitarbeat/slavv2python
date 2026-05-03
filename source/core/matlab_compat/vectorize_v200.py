"""MATLAB-shaped top-level orchestration over the maintained SLAVV processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np

from source.core.pipeline import SlavvPipeline
from source.workflows import finalize_pipeline_results


def vectorize_v200(
        image: np.ndarray,
        params: dict[str, Any],
        *,
        processor: SlavvPipeline | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        event_callback=None,
        run_dir: str | None = None,
        stop_after: str | None = None,
        force_rerun_from: str | None = None,
) -> dict[str, Any]:
    """Mirror MATLAB ``vectorize_V200`` stage order on the maintained Python pipeline."""
    active_processor = processor or SlavvPipeline()
    results = active_processor.run(
        image,
        params,
        progress_callback=progress_callback,
        event_callback=event_callback,
        run_dir=run_dir,
        stop_after=stop_after,
        force_rerun_from=force_rerun_from,
    )
    if {"energy_data", "vertices", "edges", "network"} <= results.keys():
        return results
    return finalize_pipeline_results(
        {
            "parameters": results.get("parameters", params),
            "energy_data": active_processor.energy_data,
            "vertices": active_processor.vertices,
            "edges": active_processor.edges,
            "network": active_processor.network,
        }
    )


__all__ = ["vectorize_v200"]
