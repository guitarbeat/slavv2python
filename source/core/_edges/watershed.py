"""Compatibility shim for legacy watershed edge extraction."""

from __future__ import annotations

from ..edges_internal.edge_extraction_watershed import extract_edges_watershed

__all__ = ["extract_edges_watershed"]
