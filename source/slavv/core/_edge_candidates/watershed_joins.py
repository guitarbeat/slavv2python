"""Watershed join supplementation for MATLAB frontier parity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .common import _candidate_endpoint_pair_set
from .supplement_workflow import (
    _append_supplement_row,
    _increment_origin_count,
    _merge_or_append_supplement,
    _new_supplement_payload,
)
from .watershed_candidates import _build_watershed_candidate_rows

logger = logging.getLogger(__name__)


def _supplement_matlab_frontier_candidates_with_watershed_joins(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    max_edges_per_vertex: int = 4,
    enforce_frontier_reachability: bool = False,
    require_mutual_frontier_participation: bool = False,
    parity_watershed_metric_threshold: float | None = None,
) -> dict[str, Any]:
    """Add parity-only watershed contact candidates that the local frontier misses."""
    if len(vertex_positions) < 2:
        return candidates

    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    endpoint_pair_degree_counts: dict[int, int] = {}
    for start_vertex, end_vertex in existing_pairs:
        endpoint_pair_degree_counts[int(start_vertex)] = (
            endpoint_pair_degree_counts.get(int(start_vertex), 0) + 1
        )
        endpoint_pair_degree_counts[int(end_vertex)] = (
            endpoint_pair_degree_counts.get(int(end_vertex), 0) + 1
        )

    frontier_vertices: set[int] = set()
    existing_connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2))), dtype=np.int32
    ).reshape(-1, 2)
    for start_vertex, end_vertex in existing_connections:
        if int(start_vertex) >= 0:
            frontier_vertices.add(int(start_vertex))
        if int(end_vertex) >= 0:
            frontier_vertices.add(int(end_vertex))

    candidate_rows, shared_diagnostics = _build_watershed_candidate_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        metric_threshold=parity_watershed_metric_threshold,
    )
    supplement_payload = _new_supplement_payload(
        "watershed",
        diagnostics={
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
            **shared_diagnostics,
        },
    )

    n_reachability_rejected = 0
    n_mutual_frontier_rejected = 0
    n_cap_rejected = 0
    n_endpoint_degree_rejected = 0
    n_accepted = 0
    origin_supplement_counts: dict[int, int] = {}

    for pair, trace, energy_trace, scale_trace, metric, _endpoint_distance in candidate_rows:
        if (
            enforce_frontier_reachability
            and pair[0] not in frontier_vertices
            and pair[1] not in frontier_vertices
        ):
            n_reachability_rejected += 1
            continue

        if (
            enforce_frontier_reachability
            and require_mutual_frontier_participation
            and (pair[0] not in frontier_vertices or pair[1] not in frontier_vertices)
        ):
            n_mutual_frontier_rejected += 1
            continue

        if (
            endpoint_pair_degree_counts.get(pair[0], 0) >= max_edges_per_vertex
            or endpoint_pair_degree_counts.get(pair[1], 0) >= max_edges_per_vertex
        ):
            n_endpoint_degree_rejected += 1
            continue

        seed_origin = pair[0]
        current_origin_count = origin_supplement_counts.get(seed_origin, 0)
        if current_origin_count >= max_edges_per_vertex:
            n_cap_rejected += 1
            continue

        _append_supplement_row(
            supplement_payload,
            pair=pair,
            trace=trace,
            energy_trace=energy_trace,
            scale_trace=scale_trace,
            metric=metric,
            origin_index=pair[0],
            connection_source="watershed",
        )
        supplement_payload["diagnostics"]["watershed_join_supplement_count"] += 1
        n_accepted += 1
        _increment_origin_count(
            supplement_payload["diagnostics"],
            origin_supplement_counts,
            seed_origin,
            key="watershed_per_origin_candidate_counts",
        )
        existing_pairs.add(pair)
        endpoint_pair_degree_counts[pair[0]] = endpoint_pair_degree_counts.get(pair[0], 0) + 1
        endpoint_pair_degree_counts[pair[1]] = endpoint_pair_degree_counts.get(pair[1], 0) + 1

    supplement_payload["diagnostics"]["watershed_reachability_rejected"] = n_reachability_rejected
    supplement_payload["diagnostics"]["watershed_mutual_frontier_rejected"] = (
        n_mutual_frontier_rejected
    )
    supplement_payload["diagnostics"]["watershed_cap_rejected"] = n_cap_rejected
    supplement_payload["diagnostics"]["watershed_endpoint_degree_rejected"] = (
        n_endpoint_degree_rejected
    )
    supplement_payload["diagnostics"]["watershed_accepted"] = n_accepted

    logger.info(
        "Watershed supplement: %d total pairs, %d already existing, "
        "%d reachability rejected, %d mutual-frontier rejected, "
        "%d endpoint-degree rejected, %d metric-threshold rejected, %d cap rejected, "
        "%d short-trace rejected, %d energy rejected, %d accepted",
        supplement_payload["diagnostics"]["watershed_total_pairs"],
        supplement_payload["diagnostics"]["watershed_already_existing"],
        n_reachability_rejected,
        n_mutual_frontier_rejected,
        n_endpoint_degree_rejected,
        supplement_payload["diagnostics"]["watershed_metric_threshold_rejected"],
        n_cap_rejected,
        supplement_payload["diagnostics"]["watershed_short_trace_rejected"],
        supplement_payload["diagnostics"]["watershed_energy_rejected"],
        n_accepted,
    )

    _merge_or_append_supplement(candidates, supplement_payload)
    return candidates
