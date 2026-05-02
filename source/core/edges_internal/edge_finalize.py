"""Preferred internal name for edge finalization helpers."""

from __future__ import annotations

from .._edges.postprocess import (
    finalize_edges_matlab_style,
    prefilter_edge_indices_for_cleanup_matlab_style,
)

__all__ = [
    "finalize_edges_matlab_style",
    "prefilter_edge_indices_for_cleanup_matlab_style",
]
