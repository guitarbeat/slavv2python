"""Low-level edge tracing primitives for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from slavv_python.core.edges import terminal_lookup as _lookup
from slavv_python.core.edges import trace_directions as _directions
from slavv_python.core.edges import trace_metrics as _metrics
from slavv_python.core.edges import tracing as _tracing

if TYPE_CHECKING:
    import numpy as np

TraceMetadata = _tracing.TraceMetadata
TraceEdgeResult = _tracing.TraceEdgeResult
in_bounds = _lookup.in_bounds
vertex_at_position = _lookup.vertex_at_position
near_vertex = _lookup.near_vertex
find_terminal_vertex = _lookup.find_terminal_vertex
compute_gradient = _metrics.compute_gradient
_clip_trace_indices = _metrics._clip_trace_indices
_trace_scale_series = _metrics._trace_scale_series
_trace_energy_series = _metrics._trace_energy_series
_edge_metric_from_energy_trace = _metrics._edge_metric_from_energy_trace
_record_trace_diagnostics = _metrics._record_trace_diagnostics
generate_edge_directions = _directions.generate_edge_directions
_finalize_traced_edge = _lookup._finalize_traced_edge
trace_edge = _tracing.trace_edge

__all__ = [
    "TraceEdgeResult",
    "TraceMetadata",
    "_clip_trace_indices",
    "_edge_metric_from_energy_trace",
    "_finalize_traced_edge",
    "_record_trace_diagnostics",
    "_trace_energy_series",
    "_trace_scale_series",
    "compute_gradient",
    "estimate_vessel_directions",
    "find_terminal_vertex",
    "generate_edge_directions",
    "in_bounds",
    "near_vertex",
    "trace_edge",
    "vertex_at_position",
]


def estimate_vessel_directions(
    energy: np.ndarray, pos: np.ndarray, radius: float, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Estimate vessel directions at a vertex via local Hessian analysis."""
    return cast(
        "np.ndarray",
        _directions.estimate_vessel_directions(
            energy,
            pos,
            radius,
            microns_per_voxel,
            fallback_direction_generator=generate_edge_directions,
        ),
    )
