"""Candidate manifest shaping helpers for Edge Discovery."""

from __future__ import annotations

from typing import Any

import numpy as np

from slavv_python.processing.stages.edges.edge_types import Int32Array


def normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
    *,
    default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection slavv_python label list."""
    if candidate_connection_count <= 0:
        return []

    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        source_values = []

    allowed_sources = {"frontier", "watershed", "geodesic", "fallback", "unknown"}
    default_label = default_source if default_source in allowed_sources else "unknown"
    normalized: list[str] = []
    for index in range(candidate_connection_count):
        if index < len(source_values):
            source_label = str(source_values[index]).strip().lower()
            normalized.append(source_label if source_label in allowed_sources else default_label)
            continue
        normalized.append(default_label)
    return normalized


def _candidate_endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    """Return the orientation-independent terminal endpoint pairs in a candidate payload."""
    pairs: set[tuple[int, int]] = set()
    normalized = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    for start_vertex, end_vertex in normalized:
        if int(start_vertex) < 0 or int(end_vertex) < 0:
            continue
        u, v = int(start_vertex), int(end_vertex)
        pairs.add((u, v) if u < v else (v, u))
    return pairs


def _reorder_candidate_payload(
    candidates: dict[str, Any],
    sort_order: Int32Array,
) -> dict[str, Any]:
    """Return a new candidate payload reordered by the provided sort indices."""
    if sort_order.size == 0:
        return candidates

    sort_idx = np.asarray(sort_order, dtype=np.int32).reshape(-1)

    reordered = dict(candidates)
    if "traces" in candidates:
        reordered["traces"] = [candidates["traces"][i] for i in sort_idx.tolist()]
    if "connections" in candidates:
        reordered["connections"] = np.asarray(
            candidates["connections"][sort_idx], dtype=np.int32
        ).reshape(-1, 2)
    if "metrics" in candidates:
        reordered["metrics"] = np.asarray(candidates["metrics"][sort_idx], dtype=np.float32)
    if "energy_traces" in candidates:
        reordered["energy_traces"] = [candidates["energy_traces"][i] for i in sort_idx.tolist()]
    if "scale_traces" in candidates:
        reordered["scale_traces"] = [candidates["scale_traces"][i] for i in sort_idx.tolist()]
    if "origin_indices" in candidates:
        reordered["origin_indices"] = np.asarray(
            candidates["origin_indices"][sort_idx], dtype=np.int32
        )
    if "connection_sources" in candidates:
        reordered["connection_sources"] = [
            candidates["connection_sources"][i] for i in sort_idx.tolist()
        ]

    return reordered


def _candidate_incident_pair_counts(connections: np.ndarray) -> dict[int, int]:
    """Count unique incident endpoint pairs for each vertex."""
    counts: dict[int, int] = {}
    for start_vertex, end_vertex in _candidate_endpoint_pair_set(connections):
        counts[int(start_vertex)] = counts.get(int(start_vertex), 0) + 1
        counts[int(end_vertex)] = counts.get(int(end_vertex), 0) + 1
    return counts
