"""Watershed contact augmentation for frontier candidate manifests."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._edge_payloads import _merge_edge_diagnostics
from ..edge_primitives import (
    _edge_metric_from_energy_trace,
    _trace_energy_series,
    _trace_scale_series,
)
from .audit import _normalize_candidate_connection_sources
from .candidate_manifest import _append_candidate_unit
from .common import _candidate_endpoint_pair_set
from .watershed_support import (
    _best_watershed_contact_coords,
    _build_watershed_join_trace,
    _build_watershed_labels,
)


def _augment_matlab_frontier_candidates_with_watershed_contacts(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    *,
    max_edges_per_vertex: int = 4,
    candidate_mode: str = "all_contacts",
    parity_watershed_metric_threshold: float | None = None,
) -> dict[str, Any]:
    """Merge watershed-contact candidates into the MATLAB parity frontier payload."""
    if len(vertex_positions) < 2:
        return candidates

    labels, image_shape = _build_watershed_labels(energy, vertex_positions, energy_sign)
    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    contact_coords_by_pair = _best_watershed_contact_coords(labels, energy)

    existing_connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 2)
    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        len(existing_connections),
        default_source="frontier",
    )
    origin_indices = np.asarray(
        candidates.get("origin_indices", np.zeros((0,))), dtype=np.int32
    ).reshape(-1)
    frontier_origin_counts: dict[int, int] = {}
    for index, origin_index in enumerate(origin_indices):
        if index >= len(connection_sources) or connection_sources[index] != "frontier":
            continue
        frontier_origin_counts[int(origin_index)] = (
            frontier_origin_counts.get(int(origin_index), 0) + 1
        )

    candidate_rows: list[
        tuple[float, float, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]
    ] = []
    n_already_existing = 0
    n_short_trace = 0
    n_energy_rejected = 0
    n_metric_threshold_rejected = 0
    n_total_watershed_pairs = len(contact_coords_by_pair)

    for pair, contact_coord in sorted(contact_coords_by_pair.items()):
        if pair in existing_pairs:
            n_already_existing += 1
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
                min_energy = float(np.nanmin(energy_trace_array))
                fails_metric_threshold = min_energy < parity_watershed_metric_threshold
            if fails_metric_threshold:
                n_metric_threshold_rejected += 1
                continue

        scale_trace = _trace_scale_series(trace, scale_indices)
        metric = _edge_metric_from_energy_trace(energy_trace)
        endpoint_distance = float(
            np.linalg.norm(vertex_positions[pair[0]] - vertex_positions[pair[1]])
        )
        candidate_rows.append((metric, endpoint_distance, pair, trace, energy_trace, scale_trace))

    candidate_rows.sort(key=lambda row: (row[0], row[1]))

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
            "watershed_total_pairs": n_total_watershed_pairs,
            "watershed_already_existing": n_already_existing,
            "watershed_short_trace_rejected": n_short_trace,
            "watershed_energy_rejected": n_energy_rejected,
            "watershed_metric_threshold_rejected": n_metric_threshold_rejected,
            "watershed_origin_budget_rejected": 0,
            "watershed_accepted": 0,
        },
    }
    origin_added_counts: dict[int, int] = {}
    for metric, _distance, pair, trace, energy_trace, scale_trace in candidate_rows:
        if candidate_mode == "remaining_origin_contacts":
            remaining_budget = max_edges_per_vertex - frontier_origin_counts.get(pair[0], 0)
            if remaining_budget <= 0 or origin_added_counts.get(pair[0], 0) >= remaining_budget:
                supplement_payload["diagnostics"]["watershed_origin_budget_rejected"] += 1
                continue

        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([pair[0], pair[1]])
        supplement_payload["metrics"].append(metric)
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(pair[0])
        supplement_payload["connection_sources"].append("watershed")
        supplement_payload["diagnostics"]["watershed_join_supplement_count"] += 1
        supplement_payload["diagnostics"]["watershed_accepted"] += 1
        origin_added_counts[pair[0]] = origin_added_counts.get(pair[0], 0) + 1
        supplement_payload["diagnostics"]["watershed_per_origin_candidate_counts"][str(pair[0])] = (
            origin_added_counts[pair[0]]
        )

    if supplement_payload["connections"]:
        _append_candidate_unit(candidates, supplement_payload)
    else:
        _merge_edge_diagnostics(
            candidates.get("diagnostics", {}), supplement_payload["diagnostics"]
        )
    return candidates
