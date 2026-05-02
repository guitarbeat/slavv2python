"""Preferred internal name for edge terminal lookup helpers."""

from __future__ import annotations

from .._edge_primitives.lookup import (
    find_terminal_vertex,
    in_bounds,
    near_vertex,
    vertex_at_position,
)
from .._edge_primitives.terminals import _finalize_traced_edge, _resolve_trace_terminal_vertex

__all__ = [
    "_finalize_traced_edge",
    "_resolve_trace_terminal_vertex",
    "find_terminal_vertex",
    "in_bounds",
    "near_vertex",
    "vertex_at_position",
]
