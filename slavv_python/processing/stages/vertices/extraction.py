"""Vertex extraction workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.processing.stages.vertices.manager import VertexManager

if TYPE_CHECKING:
    from slavv_python.schema.results import EnergyResult, VertexSet


def extract_vertices(energy_data: EnergyResult, params: dict[str, Any]) -> VertexSet:
    """Extract vertices as local extrema in the energy field."""
    return VertexManager.run(energy_data, params)


__all__ = ["extract_vertices"]
