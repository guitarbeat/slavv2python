"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from . import edge_extraction_standard as _standard
from . import edge_extraction_watershed as _watershed
from . import edge_units as _units
from .bridge_insertion import add_vertices_to_edges_matlab_style
from .edge_candidates import (
    _append_candidate_unit,
    _build_edge_candidate_audit,
    _build_frontier_candidate_lifecycle,
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
    _normalize_candidate_origin_counts,
    _use_matlab_frontier_tracer,
)
from .edge_finalize import finalize_edges_matlab_style
from .edge_payloads import _empty_edge_diagnostics, _empty_edges_result
from .edge_selection import choose_edges_for_workflow
from .resumable_edges import (
    extract_edges_resumable as _extract_edges_resumable,
)
from .resumable_edges import (
    extract_edges_watershed_resumable as _extract_edges_watershed_resumable,
)
from .vertices import paint_vertex_center_image, paint_vertex_image

if TYPE_CHECKING:
    from pathlib import Path

    from slavv_python.runtime import StageController


def _load_edge_units(
    units_dir: Path,
    n_vertices: int,
) -> tuple[dict[str, object], set[int]]:
    del n_vertices
    return cast(
        "tuple[dict[str, object], set[int]]",
        _units._load_edge_units(units_dir, _append_candidate_unit, _empty_edge_diagnostics),
    )


def extract_edges_watershed(
    energy_data: dict[str, object], vertices: dict[str, object], params: dict[str, object]
) -> dict[str, object]:
    """Extract edges using watershed segmentation seeded at vertices."""
    return cast(
        "dict[str, object]",
        _watershed.extract_edges_watershed(energy_data, vertices, params),
    )


def extract_edges(
    energy_data: dict[str, object], vertices: dict[str, object], params: dict[str, object]
) -> dict[str, object]:
    """Extract edges by tracing from vertices through energy field."""
    return cast(
        "dict[str, object]",
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
    energy_data: dict[str, object],
    vertices: dict[str, object],
    params: dict[str, object],
    stage_controller: StageController,
) -> dict[str, object]:
    """Trace edges with per-origin persisted units."""
    from slavv_python.runtime.run_state import atomic_joblib_dump

    return cast(
        "dict[str, object]",
        _extract_edges_resumable(
            energy_data,
            vertices,
            params,
            stage_controller,
            atomic_joblib_dump=atomic_joblib_dump,
            empty_edges_result=_empty_edges_result,
            build_edge_candidate_audit=_build_edge_candidate_audit,
            build_frontier_candidate_lifecycle=_build_frontier_candidate_lifecycle,
            finalize_matlab_parity_candidates=_finalize_matlab_parity_candidates,
            normalize_candidate_origin_counts=_normalize_candidate_origin_counts,
            generate_edge_candidates_matlab_frontier=_generate_edge_candidates_matlab_frontier,
            generate_edge_candidates=_generate_edge_candidates,
            choose_edges_for_workflow=choose_edges_for_workflow,
            add_vertices_to_edges_matlab_style=add_vertices_to_edges_matlab_style,
            finalize_edges_matlab_style=finalize_edges_matlab_style,
            paint_vertex_center_image=paint_vertex_center_image,
            paint_vertex_image=paint_vertex_image,
            use_matlab_frontier_tracer=_use_matlab_frontier_tracer,
        ),
    )


def extract_edges_watershed_resumable(
    energy_data: dict[str, object],
    vertices: dict[str, object],
    params: dict[str, object],
    stage_controller: StageController,
) -> dict[str, object]:
    """Extract watershed edges with per-label persisted units."""
    from slavv_python.runtime.run_state import atomic_joblib_dump

    return cast(
        "dict[str, object]",
        _extract_edges_watershed_resumable(
            energy_data,
            vertices,
            params,
            stage_controller=stage_controller,
            atomic_joblib_dump=atomic_joblib_dump,
            append_candidate_unit=_append_candidate_unit,
            empty_edge_diagnostics=_empty_edge_diagnostics,
        ),
    )
