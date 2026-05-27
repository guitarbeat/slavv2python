"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from slavv_python.processing.stages.edges import extraction_standard as _standard
from slavv_python.processing.stages.edges import extraction_watershed as _watershed
from slavv_python.processing.stages.edges import units as _units
from slavv_python.processing.stages.edges.bridge_insertion import add_vertices_to_edges_matlab_style
from slavv_python.processing.stages.edges.candidate_generation import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from slavv_python.processing.stages.edges.common import _use_matlab_frontier_tracer
from slavv_python.processing.stages.edges.finalize import finalize_edges_matlab_style
from slavv_python.processing.stages.edges.manager import EdgeManager
from slavv_python.processing.stages.edges.payloads import (
    _empty_edge_diagnostics,
    _empty_edges_result,
)
from slavv_python.processing.stages.edges.selection import choose_edges_for_workflow
from slavv_python.processing.stages.vertices import paint_vertex_center_image, paint_vertex_image

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
    return cast(
        "EdgeSet",
        _standard.extract_edges(
            energy_data,
            vertices,
            params,
            empty_edges_result=_empty_edges_result,
            paint_vertex_center_image=paint_vertex_center_image,
            paint_vertex_image=paint_vertex_image,
            use_matlab_frontier_tracer=_use_matlab_frontier_tracer,
            generate_edge_candidates_matlab_frontier=_generate_edge_candidates_matlab_frontier,
            finalize_matlab_parity_candidates=_finalize_matlab_parity_candidates,
            generate_edge_candidates=_generate_edge_candidates,
            choose_edges_for_workflow=choose_edges_for_workflow,
            add_vertices_to_edges_matlab_style=add_vertices_to_edges_matlab_style,
            finalize_edges_matlab_style=finalize_edges_matlab_style,
        ),
    )


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
