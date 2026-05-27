"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from slavv_python.processing.stages.edges import extraction_watershed as _watershed
from slavv_python.processing.stages.edges import units as _units
from slavv_python.processing.stages.edges.manager import EdgeManager
from slavv_python.processing.stages.edges.payloads import _empty_edge_diagnostics

if TYPE_CHECKING:
    from pathlib import Path

    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet


def _load_edge_units(
    units_dir: Path,
    n_vertices: int,
) -> tuple[dict[str, object], set[int]]:
    del n_vertices
    return cast(
        "tuple[dict[str, object], set[int]]",
        _units._load_edge_units(units_dir, None, _empty_edge_diagnostics),
    )


def extract_edges_watershed(
    energy_data: EnergyResult, vertices: VertexSet, params: dict[str, Any]
) -> EdgeSet:
    """Extract edges using watershed segmentation seeded at vertices."""
    return cast(
        "EdgeSet",
        _watershed.extract_edges_watershed(energy_data, vertices, params),
    )


def extract_edges(
    energy_data: EnergyResult, vertices: VertexSet, params: dict[str, Any]
) -> EdgeSet:
    """Extract edges by tracing from vertices through energy field."""
    return EdgeManager.run(energy_data, vertices, params)


def extract_edges_resumable(
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    stage_controller: StageController,
) -> EdgeSet:
    """Trace edges with checkpointed candidate generation and selection."""
    return EdgeManager.run_resumable(energy_data, vertices, params, stage_controller)


def extract_edges_watershed_resumable(
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    stage_controller: StageController,
) -> EdgeSet:
    """Extract watershed edges with per-label persisted units."""
    return EdgeManager.run_watershed_resumable(energy_data, vertices, params, stage_controller)
