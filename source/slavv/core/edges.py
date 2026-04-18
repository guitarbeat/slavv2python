"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ._edge_payloads import _empty_edge_diagnostics, _empty_edges_result
from ._edges import resumable as _resumable
from ._edges import standard as _standard
from ._edges import units as _units
from ._edges import watershed as _watershed
from ._radius_utils import _scalar_radius
from .edge_candidates import (
    _append_candidate_unit,
    _build_edge_candidate_audit,
    _build_frontier_candidate_lifecycle,
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
    _normalize_candidate_origin_counts,
    _trace_origin_edges_matlab_frontier,
    _use_matlab_frontier_tracer,
)
from .edge_primitives import (
    _edge_metric_from_energy_trace,
    _record_trace_diagnostics,
    _trace_energy_series,
    _trace_scale_series,
    estimate_vessel_directions,
    generate_edge_directions,
    trace_edge,
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
            use_matlab_frontier_tracer=_use_matlab_frontier_tracer,
            generate_edge_candidates_matlab_frontier=_generate_edge_candidates_matlab_frontier,
            finalize_matlab_parity_candidates=_finalize_matlab_parity_candidates,
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
    from slavv.runtime.run_state import atomic_joblib_dump, atomic_write_json

    return cast(
        "dict[str, object]",
        _resumable.extract_edges_resumable(
            energy_data,
            vertices,
            params,
            stage_controller,
            atomic_joblib_dump=atomic_joblib_dump,
            atomic_write_json=atomic_write_json,
            empty_edges_result=_empty_edges_result,
            empty_edge_diagnostics=_empty_edge_diagnostics,
            scalar_radius=_scalar_radius,
            append_candidate_unit=_append_candidate_unit,
            build_edge_candidate_audit=_build_edge_candidate_audit,
            build_frontier_candidate_lifecycle=_build_frontier_candidate_lifecycle,
            finalize_matlab_parity_candidates=_finalize_matlab_parity_candidates,
            normalize_candidate_origin_counts=_normalize_candidate_origin_counts,
            trace_origin_edges_matlab_frontier=_trace_origin_edges_matlab_frontier,
            use_matlab_frontier_tracer=_use_matlab_frontier_tracer,
            edge_metric_from_energy_trace=_edge_metric_from_energy_trace,
            record_trace_diagnostics=_record_trace_diagnostics,
            trace_energy_series=_trace_energy_series,
            trace_scale_series=_trace_scale_series,
            estimate_vessel_directions=estimate_vessel_directions,
            generate_edge_directions=generate_edge_directions,
            trace_edge=trace_edge,
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
