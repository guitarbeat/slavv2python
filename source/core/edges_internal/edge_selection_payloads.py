"""Shared payload and preprocessing helpers for edge selection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .._edge_candidates.common import (
    normalize_candidate_connection_sources as _normalize_connection_sources,
)
from .._edge_payloads import _empty_edges_result, build_edge_diagnostics


def normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
    *,
    default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection source label list."""
    return _normalize_connection_sources(
        raw_sources,
        candidate_connection_count,
        default_source=default_source,
    )


def empty_edge_diagnostics() -> dict[str, Any]:
    """Return the canonical edge-diagnostics payload."""
    diagnostics = build_edge_diagnostics(extra_fields={})
    return cast("dict[str, Any]", diagnostics)


def initialize_edge_selection_diagnostics(
    candidates: dict[str, Any],
    connections: np.ndarray,
    traces: list[np.ndarray],
) -> dict[str, Any]:
    """Initialize chooser diagnostics from candidate-stage counters and counts."""
    diagnostics = empty_edge_diagnostics()
    candidate_diagnostics = candidates.get("diagnostics", {})
    for key, value in candidate_diagnostics.items():
        diagnostics[key] = value.copy() if key == "stop_reason_counts" else value
    diagnostics["candidate_traced_edge_count"] = len(traces)
    diagnostics["terminal_edge_count"] = (
        int(np.sum(connections[:, 1] >= 0)) if len(connections) else 0
    )
    diagnostics["self_edge_count"] = (
        int(np.sum(connections[:, 0] == connections[:, 1])) if len(connections) else 0
    )
    diagnostics["dangling_edge_count"] = (
        int(np.sum(connections[:, 1] < 0)) if len(connections) else 0
    )
    return diagnostics


def prepare_candidate_indices_for_cleanup(
    connections: np.ndarray,
    metrics: np.ndarray,
    energy_traces: list[np.ndarray],
    diagnostics: dict[str, Any],
    *,
    subset_indices: list[int] | np.ndarray | None = None,
    reject_nonnegative_energy_edges: bool = True,
) -> list[int]:
    """Apply MATLAB ``clean_edge_pairs`` ordering before downstream cleanup."""
    if subset_indices is not None:
        base_indices = np.asarray(subset_indices, dtype=np.int32)
    else:
        base_indices = np.arange(len(connections), dtype=np.int32)

    if base_indices.size == 0:
        return []

    subset_connections = connections[base_indices]
    valid = (subset_connections[:, 0] != subset_connections[:, 1]) & (subset_connections[:, 1] >= 0)
    filtered_indices = base_indices[np.flatnonzero(valid)]

    if filtered_indices.size and reject_nonnegative_energy_edges:
        nonnegative_max = np.array(
            [
                np.nanmax(np.asarray(energy_traces[index], dtype=np.float32)) >= 0
                for index in filtered_indices
            ],
            dtype=bool,
        )
        diagnostics["negative_energy_rejected_count"] = int(
            diagnostics.get("negative_energy_rejected_count", 0)
        ) + int(np.sum(nonnegative_max))
        filtered_indices = filtered_indices[~nonnegative_max]

    if filtered_indices.size == 0:
        return []

    edge_lengths = np.asarray(
        [len(np.asarray(energy_traces[index], dtype=np.float32)) for index in filtered_indices],
        dtype=np.int32,
    )
    ordered = filtered_indices[np.argsort(edge_lengths, kind="stable")]
    ordered = ordered[np.argsort(metrics[ordered], kind="stable")]

    directed_seen: set[tuple[int, int]] = set()
    directed_indices: list[int] = []
    for index in ordered:
        pair_d = (int(connections[index, 0]), int(connections[index, 1]))
        if pair_d in directed_seen:
            diagnostics["duplicate_directed_pair_count"] += 1
            continue
        directed_seen.add(pair_d)
        directed_indices.append(int(index))

    undirected_seen: set[tuple[int, int]] = set()
    filtered_unique_indices: list[int] = []
    for index in directed_indices:
        start_vertex, end_vertex = int(connections[index, 0]), int(connections[index, 1])
        pair_u = (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )
        if pair_u in undirected_seen:
            diagnostics["antiparallel_pair_count"] += 1
            continue
        undirected_seen.add(pair_u)
        filtered_unique_indices.append(int(index))

    return filtered_unique_indices


def build_selected_edges_result(
    final_indices: list[int],
    traces: list[np.ndarray],
    connections: np.ndarray,
    metrics: np.ndarray,
    energy_traces: list[np.ndarray],
    scale_traces: list[np.ndarray],
    connection_sources: list[str],
    vertex_positions: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical chosen-edge payload from candidate indices."""
    result = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
    result["traces"] = [np.asarray(traces[index], dtype=np.float32) for index in final_indices]
    result["connections"] = np.asarray(connections[final_indices], dtype=np.int32).reshape(-1, 2)
    result["energies"] = np.asarray(metrics[final_indices], dtype=np.float32)
    result["energy_traces"] = [
        np.asarray(energy_traces[index], dtype=np.float32) for index in final_indices
    ]
    result["scale_traces"] = [
        np.asarray(scale_traces[index], dtype=np.int16) for index in final_indices
    ]
    result["connection_sources"] = [
        connection_sources[index] if index < len(connection_sources) else "unknown"
        for index in final_indices
    ]
    result["chosen_candidate_indices"] = np.asarray(final_indices, dtype=np.int32)
    diagnostics["chosen_edge_count"] = len(final_indices)
    result["diagnostics"] = diagnostics
    return result


__all__ = [
    "build_selected_edges_result",
    "empty_edge_diagnostics",
    "initialize_edge_selection_diagnostics",
    "normalize_candidate_connection_sources",
    "prepare_candidate_indices_for_cleanup",
]
