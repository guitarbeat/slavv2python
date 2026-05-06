"""Compatibility shim for legacy post-choice edge postprocessing."""

from __future__ import annotations

from ..edges_internal.edge_finalize import (
    _matlab_crop_edges_v200,
    _matlab_edge_endpoint_energy,
    finalize_edges_matlab_style,
    normalize_edges_matlab_style,
    prefilter_edge_indices_for_cleanup_matlab_style,
)

__all__ = [
    "_matlab_crop_edges_v200",
    "_matlab_edge_endpoint_energy",
    "finalize_edges_matlab_style",
    "normalize_edges_matlab_style",
    "prefilter_edge_indices_for_cleanup_matlab_style",
]
