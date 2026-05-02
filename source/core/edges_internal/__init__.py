"""Preferred internal edge package names for SLAVV."""

from __future__ import annotations

from .bridge_insertion import add_vertices_to_edges_matlab_style
from .candidate_generation import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from .edge_finalize import finalize_edges_matlab_style
from .edge_selection import choose_edges_for_workflow
from .edge_tracing import TraceEdgeResult, TraceMetadata, trace_edge
from .matlab_frontier import _generate_edge_candidates_matlab_global_watershed
from .resumable_edges import extract_edges_resumable
from .terminal_lookup import find_terminal_vertex, in_bounds, near_vertex, vertex_at_position
from .trace_directions import estimate_vessel_directions, generate_edge_directions
from .trace_metrics import compute_gradient

__all__ = [
    "TraceEdgeResult",
    "TraceMetadata",
    "_finalize_matlab_parity_candidates",
    "_generate_edge_candidates",
    "_generate_edge_candidates_matlab_frontier",
    "_generate_edge_candidates_matlab_global_watershed",
    "add_vertices_to_edges_matlab_style",
    "choose_edges_for_workflow",
    "compute_gradient",
    "estimate_vessel_directions",
    "extract_edges_resumable",
    "finalize_edges_matlab_style",
    "find_terminal_vertex",
    "generate_edge_directions",
    "in_bounds",
    "near_vertex",
    "trace_edge",
    "vertex_at_position",
]
