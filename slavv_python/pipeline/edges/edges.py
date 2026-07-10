"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from slavv_python.pipeline.edges import extraction_watershed as _watershed
from slavv_python.pipeline.edges import units as _units
from slavv_python.pipeline.edges.manager import EdgeManager
from slavv_python.pipeline.edges.payloads import _empty_edge_diagnostics

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
    """Extract edges via skimage label adjacency (NOT Certification Watershed Discovery).

    Prefer ``EdgeManager`` / ``WatershedDiscovery`` for Exact Route parity and
    ADR 0012. This helper seeds skimage watershed markers and links adjacent
    labels — a different algorithm from MATLAB global Watershed Discovery.

    Args:
        energy_data: The multi-scale energy field result.
        vertices: The set of accepted vertices to use as seeds.
        params: Authoritative configuration dictionary.

    Returns:
        An EdgeSet object containing segments and intensities.
    """
    return cast(
        "EdgeSet",
        _watershed.extract_edges_watershed(energy_data, vertices, params),
    )


def extract_edges(
    energy_data: EnergyResult, vertices: VertexSet, params: dict[str, Any]
) -> EdgeSet:
    """Extract vascular edges by tracing local minima between vertices.

    Args:
        energy_data: The multi-scale energy field result.
        vertices: The set of accepted vertices to use as seeds.
        params: Authoritative configuration dictionary.

    Returns:
        An EdgeSet object containing traced centerlines and intensities.
    """
    return EdgeManager.run(energy_data, vertices, params)


def extract_edges_resumable(
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    stage_controller: StageController,
) -> EdgeSet:
    """Extract edges with checkpointed candidate generation and selection.

    Args:
        energy_data: The multi-scale energy field result.
        vertices: The set of accepted vertices to use as seeds.
        params: Authoritative configuration dictionary.
        stage_controller: Controller for resumable checkpointing.

    Returns:
        An EdgeSet object containing segments and intensities.
    """
    return EdgeManager.run_resumable(energy_data, vertices, params, stage_controller)


def extract_edges_watershed_resumable(
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    stage_controller: StageController,
) -> EdgeSet:
    """Extract watershed edges with per-label persisted units.

    Args:
        energy_data: The multi-scale energy field result.
        vertices: The set of accepted vertices to use as seeds.
        params: Authoritative configuration dictionary.
        stage_controller: Controller for resumable checkpointing.

    Returns:
        An EdgeSet object containing segments and intensities.
    """
    return EdgeManager.run_watershed_resumable(energy_data, vertices, params, stage_controller)
