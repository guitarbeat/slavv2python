"""Compatibility shim for legacy bridge vertices insertion."""

from __future__ import annotations

from ..edges_internal.bridge_insertion import (
    _matlab_bridge_search_target,
    add_vertices_to_edges_matlab_style,
)

__all__ = [
    "_matlab_bridge_search_target",
    "add_vertices_to_edges_matlab_style",
]
