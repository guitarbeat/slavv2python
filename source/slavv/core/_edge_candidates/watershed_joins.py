"""Watershed join supplementation for MATLAB frontier parity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._edge_payloads import _merge_edge_diagnostics
from ..edge_primitives import (
    _edge_metric_from_energy_trace,
    _trace_energy_series,
    _trace_scale_series,
)
from .candidate_manifest import _append_candidate_unit
from .common import _candidate_endpoint_pair_set
from .watershed_support import (
    _best_watershed_contact_coords,
    _build_watershed_join_trace,
    _build_watershed_labels,
)

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

    labels, image_shape = _build_watershed_labels(energy, vertex_positions, energy_sign)
    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    contact_coords_by_pair = _best_watershed_contact_coords(labels, energy)
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

    supplement_payload: dict[str, Any] = {
        "candidate_source": "watershed",
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
        },
    }

    n_already_existing = 0
    n_short_trace = 0
    n_energy_rejected = 0
    n_reachability_rejected = 0
    n_mutual_frontier_rejected = 0
    n_cap_rejected = 0
    n_endpoint_degree_rejected = 0
    n_metric_threshold_rejected = 0
    n_accepted = 0
    n_total_watershed_pairs = len(contact_coords_by_pair)
    origin_supplement_counts: dict[int, int] = {}

    for pair, contact_coord in sorted(contact_coords_by_pair.items()):
        if pair in existing_pairs:
            n_already_existing += 1
            continue

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

        trace = _build_watershed_join_trace(
            vertex_positions[pair[0]],
            contact_coord,
            vertex_positions[pair[1]],
            image_shape,
        )
        if len(trace) <= 1:
            n_short_trace += 1
            continue

        energy_trace = _trace_energy_series(trace, energy)
        energy_trace_array = np.asarray(energy_trace, dtype=np.float32)
        max_energy = float(np.nanmax(energy_trace_array))
        if energy_sign < 0:
            is_invalid = max_energy >= 0
        else:
            min_energy = float(np.nanmin(energy_trace_array))
            is_invalid = min_energy <= 0

        if is_invalid:
            n_energy_rejected += 1
            continue

        if parity_watershed_metric_threshold is not None:
            if energy_sign < 0:
                fails_metric_threshold = max_energy > parity_watershed_metric_threshold
            else:
                fails_metric_threshold = min_energy < parity_watershed_metric_threshold
            if fails_metric_threshold:
                n_metric_threshold_rejected += 1
                continue

        scale_trace = _trace_scale_series(trace, scale_indices)
        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([pair[0], pair[1]])
        supplement_payload["metrics"].append(_edge_metric_from_energy_trace(energy_trace))
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(pair[0])
        supplement_payload["connection_sources"].append("watershed")
        supplement_payload["diagnostics"]["watershed_join_supplement_count"] += 1
        n_accepted += 1
        origin_supplement_counts[seed_origin] = current_origin_count + 1
        existing_pairs.add(pair)
        endpoint_pair_degree_counts[pair[0]] = endpoint_pair_degree_counts.get(pair[0], 0) + 1
        endpoint_pair_degree_counts[pair[1]] = endpoint_pair_degree_counts.get(pair[1], 0) + 1
        supplement_payload["diagnostics"]["watershed_per_origin_candidate_counts"][
            str(seed_origin)
        ] = int(origin_supplement_counts.get(seed_origin, 0))

    supplement_payload["diagnostics"]["watershed_total_pairs"] = n_total_watershed_pairs
    supplement_payload["diagnostics"]["watershed_already_existing"] = n_already_existing
    supplement_payload["diagnostics"]["watershed_short_trace_rejected"] = n_short_trace
    supplement_payload["diagnostics"]["watershed_energy_rejected"] = n_energy_rejected
    supplement_payload["diagnostics"]["watershed_reachability_rejected"] = n_reachability_rejected
    supplement_payload["diagnostics"]["watershed_mutual_frontier_rejected"] = (
        n_mutual_frontier_rejected
    )
    supplement_payload["diagnostics"]["watershed_cap_rejected"] = n_cap_rejected
    supplement_payload["diagnostics"]["watershed_endpoint_degree_rejected"] = (
        n_endpoint_degree_rejected
    )
    supplement_payload["diagnostics"]["watershed_metric_threshold_rejected"] = (
        n_metric_threshold_rejected
    )
    supplement_payload["diagnostics"]["watershed_accepted"] = n_accepted

    logger.info(
        "Watershed supplement: %d total pairs, %d already existing, "
        "%d reachability rejected, %d mutual-frontier rejected, "
        "%d endpoint-degree rejected, %d metric-threshold rejected, %d cap rejected, "
        "%d short-trace rejected, %d energy rejected, %d accepted",
        n_total_watershed_pairs,
        n_already_existing,
        n_reachability_rejected,
        n_mutual_frontier_rejected,
        n_endpoint_degree_rejected,
        n_metric_threshold_rejected,
        n_cap_rejected,
        n_short_trace,
        n_energy_rejected,
        n_accepted,
    )

    if supplement_payload["connections"]:
        _append_candidate_unit(candidates, supplement_payload)
    else:
        _merge_edge_diagnostics(
            candidates.get("diagnostics", {}), supplement_payload["diagnostics"]
        )
    return candidates
