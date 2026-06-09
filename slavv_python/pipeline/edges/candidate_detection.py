"""Backward-compatible re-exports; vertex detection lives in ``vertices.detection``."""

from __future__ import annotations

from slavv_python.pipeline.vertices.detection import (
    choose_vertices_matlab_style,
    chunk_lattice_dimensions,
    crop_vertices_matlab_style,
    ellipsoid_offsets,
    iter_overlapping_chunks,
    matlab_vertex_candidates,
    matlab_vertex_candidates_in_chunk,
    vertex_neighborhood_slices,
    vertex_window_apothem,
)

__all__ = [
    "choose_vertices_matlab_style",
    "chunk_lattice_dimensions",
    "crop_vertices_matlab_style",
    "ellipsoid_offsets",
    "iter_overlapping_chunks",
    "matlab_vertex_candidates",
    "matlab_vertex_candidates_in_chunk",
    "vertex_neighborhood_slices",
    "vertex_window_apothem",
]
