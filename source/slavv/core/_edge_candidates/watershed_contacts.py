"""Watershed contact augmentation for frontier candidate manifests."""

from __future__ import annotations

from typing import Any

import numpy as np

from .audit import _normalize_candidate_connection_sources
from .supplement_workflow import (
    _append_supplement_row,
    _increment_origin_count,
    _merge_or_append_supplement,
    _new_supplement_payload,
)
from .watershed_candidates import _build_watershed_candidate_rows


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

    candidate_rows, shared_diagnostics = _build_watershed_candidate_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        metric_threshold=parity_watershed_metric_threshold,
    )
    candidate_rows.sort(key=lambda row: (row[4], row[5]))

    supplement_payload = _new_supplement_payload(
        "watershed",
        diagnostics={
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
            **shared_diagnostics,
            "watershed_origin_budget_rejected": 0,
            "watershed_accepted": 0,
        },
    )
    origin_added_counts: dict[int, int] = {}
    for pair, trace, energy_trace, scale_trace, metric, _distance in candidate_rows:
        if candidate_mode == "remaining_origin_contacts":
            remaining_budget = max_edges_per_vertex - frontier_origin_counts.get(pair[0], 0)
            if remaining_budget <= 0 or origin_added_counts.get(pair[0], 0) >= remaining_budget:
                supplement_payload["diagnostics"]["watershed_origin_budget_rejected"] += 1
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
        supplement_payload["diagnostics"]["watershed_accepted"] += 1
        _increment_origin_count(
            supplement_payload["diagnostics"],
            origin_added_counts,
            pair[0],
            key="watershed_per_origin_candidate_counts",
        )

    _merge_or_append_supplement(candidates, supplement_payload)
    return candidates
