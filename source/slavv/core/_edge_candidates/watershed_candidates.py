"""Shared watershed candidate row construction for parity supplementation."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from ..edge_primitives import (
    _edge_metric_from_energy_trace,
    _trace_energy_series,
    _trace_scale_series,
)
from .common import _candidate_endpoint_pair_set
from .watershed_support import (
    _best_watershed_contact_coords,
    _build_watershed_join_trace,
    _build_watershed_labels,
)

WatershedCandidateRow: TypeAlias = tuple[
    tuple[int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]


def _build_watershed_candidate_rows(
    candidates: dict[str, object],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    parity_watershed_metric_threshold: float | None = None,
) -> tuple[list[WatershedCandidateRow], dict[str, int]]:
    """Build shared watershed candidate rows before policy-specific filtering."""
    if len(vertex_positions) < 2:
        return [], _empty_watershed_row_diagnostics()

    labels, image_shape = _build_watershed_labels(energy, vertex_positions, energy_sign)
    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    contact_coords_by_pair = _best_watershed_contact_coords(labels, energy)

    rows: list[WatershedCandidateRow] = []
    diagnostics = _empty_watershed_row_diagnostics()
    diagnostics["watershed_total_pairs"] = len(contact_coords_by_pair)

    for pair, contact_coord in sorted(contact_coords_by_pair.items()):
        if pair in existing_pairs:
            diagnostics["watershed_already_existing"] += 1
            continue

        trace = _build_watershed_join_trace(
            vertex_positions[pair[0]],
            contact_coord,
            vertex_positions[pair[1]],
            image_shape,
        )
        if len(trace) <= 1:
            diagnostics["watershed_short_trace_rejected"] += 1
            continue

        energy_trace = _trace_energy_series(trace, energy)
        energy_trace_array = np.asarray(energy_trace, dtype=np.float32)
        max_energy = float(np.nanmax(energy_trace_array))
        min_energy = float(np.nanmin(energy_trace_array))
        if energy_sign < 0:
            is_invalid = max_energy >= 0
        else:
            is_invalid = min_energy <= 0
        if is_invalid:
            diagnostics["watershed_energy_rejected"] += 1
            continue

        if parity_watershed_metric_threshold is not None:
            if energy_sign < 0:
                fails_metric_threshold = max_energy > parity_watershed_metric_threshold
            else:
                fails_metric_threshold = min_energy < parity_watershed_metric_threshold
            if fails_metric_threshold:
                diagnostics["watershed_metric_threshold_rejected"] += 1
                continue

        scale_trace = _trace_scale_series(trace, scale_indices)
        metric = _edge_metric_from_energy_trace(energy_trace)
        endpoint_distance = float(
            np.linalg.norm(vertex_positions[pair[0]] - vertex_positions[pair[1]])
        )
        rows.append((pair, trace, energy_trace, scale_trace, metric, endpoint_distance))

    return rows, diagnostics


def _empty_watershed_row_diagnostics() -> dict[str, int]:
    """Return the shared watershed candidate-row diagnostic counters."""
    return {
        "watershed_total_pairs": 0,
        "watershed_already_existing": 0,
        "watershed_short_trace_rejected": 0,
        "watershed_energy_rejected": 0,
        "watershed_metric_threshold_rejected": 0,
    }
