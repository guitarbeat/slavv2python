"""Count normalization helpers for parity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _coerce_count(value: Any) -> int | None:
    """Convert a count-like value into an integer when possible."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.size == 1:
            return int(np.asarray(value).item())
        return int(value.shape[0])
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_items(value: Any) -> int:
    """Count rows/items for numpy arrays and sequence-like containers."""
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return 1 if value.ndim == 0 else int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 1


def _resolve_count(explicit: Any, inferred: int) -> int:
    """Prefer explicit counts when present, otherwise fall back to inferred ones."""
    explicit_count = _coerce_count(explicit)
    if explicit_count is not None and (explicit_count > 0 or inferred == 0):
        return explicit_count
    return inferred


def _infer_vertices_count(vertices: dict[str, Any]) -> int:
    """Infer vertex count from payload structure when `count` is absent."""
    if not isinstance(vertices, dict):
        return 0
    return _resolve_count(vertices.get("count"), _count_items(vertices.get("positions")))


def _infer_edges_count(edges: dict[str, Any]) -> int:
    """Infer edge count from connections or traces when `count` is absent."""
    if not isinstance(edges, dict):
        return 0
    inferred = _count_items(edges.get("connections"))
    if inferred == 0:
        inferred = _count_items(edges.get("traces"))
    return _resolve_count(edges.get("count"), inferred)


def _infer_strand_count(network: dict[str, Any]) -> int:
    """Infer strand count from network topology when explicit counts are absent."""
    if not isinstance(network, dict):
        return 0
    return _resolve_count(network.get("strand_count"), _count_items(network.get("strands")))
