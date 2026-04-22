"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ._edge_payloads import _empty_edge_diagnostics, _empty_edges_result
from ._edges import resumable as _resumable
from ._edges import standard as _standard
from ._edges import units as _units
from ._edges import watershed as _watershed
from .edge_candidates import (
    _append_candidate_unit,
    _build_edge_candidate_audit,
    _generate_edge_candidates,
    _normalize_candidate_origin_counts,
)
from .edge_selection import choose_edges_for_workflow
from .vertices import paint_vertex_center_image

if TYPE_CHECKING:
    from pathlib import Path

    from slavv.runtime import StageController


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
            generate_edge_candidates=_generate_edge_candidates,
            choose_edges_for_workflow=choose_edges_for_workflow,
        ),
    )


def extract_edges_resumable(
    energy_data: dict[str, object],
    vertices: dict[str, object],
    params: dict[str, object],
    stage_controller: StageController,
) -> dict[str, object]:
    """Trace edges with per-origin persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump

    return cast(
        "dict[str, object]",
        _resumable.extract_edges_resumable(
            energy_data,
            vertices,
            params,
            stage_controller,
            atomic_joblib_dump=atomic_joblib_dump,
            empty_edges_result=_empty_edges_result,
            build_edge_candidate_audit=_build_edge_candidate_audit,
            normalize_candidate_origin_counts=_normalize_candidate_origin_counts,
            generate_edge_candidates=_generate_edge_candidates,
            choose_edges_for_workflow=choose_edges_for_workflow,
            paint_vertex_center_image=paint_vertex_center_image,
        ),
    )


def extract_edges_watershed_resumable(
    energy_data: dict[str, object],
    vertices: dict[str, object],
    params: dict[str, object],
    stage_controller: StageController,
) -> dict[str, object]:
    """Extract watershed edges with per-label persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump

    return cast(
        "dict[str, object]",
        _resumable.extract_edges_watershed_resumable(
            energy_data,
            vertices,
            params,
            stage_controller=stage_controller,
            atomic_joblib_dump=atomic_joblib_dump,
            append_candidate_unit=_append_candidate_unit,
            empty_edge_diagnostics=_empty_edge_diagnostics,
        ),
    )
