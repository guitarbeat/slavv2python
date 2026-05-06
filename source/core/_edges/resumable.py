"""Compatibility shim for legacy resumable edge extraction."""

from __future__ import annotations

from ..edges_internal.resumable_edges import (
    extract_edges_resumable,
    extract_edges_watershed_resumable,
)

__all__ = [
    "extract_edges_resumable",
    "extract_edges_watershed_resumable",
]
