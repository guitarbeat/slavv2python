"""Resumable vertex extraction workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.processing.stages.vertices.manager import VertexManager

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult, VertexSet


def extract_vertices_resumable(
    energy_data: EnergyResult,
    params: dict[str, Any],
    stage_controller: StageController,
) -> VertexSet:
    """Extract vertices with persisted MATLAB-style scan, crop, and choose state."""
    return VertexManager.run_resumable(energy_data, params, stage_controller)


__all__ = ["extract_vertices_resumable"]
