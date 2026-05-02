"""Preferred internal name for resumable edge extraction."""

from __future__ import annotations

from .._edges.resumable import extract_edges_resumable, extract_edges_watershed_resumable

__all__ = [
    "extract_edges_resumable",
    "extract_edges_watershed_resumable",
]
