"""Shared watershed candidate row construction for candidate augmentation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from typing_extensions import TypeAlias

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

WatershedCandidateRow: TypeAlias = tuple[
    tuple[int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]

_ALLOWED_WATERSHED_CANDIDATE_MODES = {
    "all_contacts",
    "remaining_origin_contacts",
    "origin_cap",
}


def _parity_watershed_candidate_mode(params: dict[str, Any]) -> str | None:
    """Return the configured watershed candidate mode for exact-network reruns."""
    requested_mode = params.get(
        "parity_watershed_candidate_mode",
        params.get("watershed_candidate_mode"),
    )
    if requested_mode in (None, ""):
        if not bool(params.get("comparison_exact_network", False)):
            return None
        requested_mode = "all_contacts"

    normalized_mode = str(requested_mode).strip().lower()
    if normalized_mode not in _ALLOWED_WATERSHED_CANDIDATE_MODES:
        return "all_contacts"
    return normalized_mode


def _parity_watershed_metric_threshold_from_params(
    params: dict[str, Any],
) -> float | None:
    """Return the optional watershed metric threshold used for parity runs."""
    threshold_raw = params.get(
        "parity_watershed_metric_threshold",
        params.get("watershed_metric_threshold"),
    )
    if threshold_raw in (None, ""):
        return None
    return float(cast("Any", threshold_raw))


def _build_watershed_candidate_rows(
    candidates: dict[str, object],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    metric_threshold: float | None = None,
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

        if metric_threshold is not None:
            if energy_sign < 0:
                fails_metric_threshold = max_energy > metric_threshold
            else:
                fails_metric_threshold = min_energy < metric_threshold
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


def _augment_candidates_with_watershed_contacts(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    *,
    max_edges_per_vertex: int = 4,
    candidate_mode: str = "all_contacts",
    metric_threshold: float | None = None,
) -> dict[str, Any]:
    """Append watershed-contact candidates into an existing candidate manifest."""
    if len(vertex_positions) < 2:
        return candidates

    candidate_rows, shared_diagnostics = _build_watershed_candidate_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        metric_threshold=metric_threshold,
    )
    if not candidate_rows:
        diagnostics = candidates.setdefault("diagnostics", {})
        for key, value in shared_diagnostics.items():
            diagnostics[key] = int(diagnostics.get(key, 0)) + int(value)
        return candidates

    origin_indices = np.asarray(
        candidates.get("origin_indices", np.zeros((0,), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    existing_origin_counts: dict[int, int] = {}
    for origin_index in origin_indices:
        origin_index_int = int(origin_index)
        if origin_index_int < 0:
            continue
        existing_origin_counts[origin_index_int] = existing_origin_counts.get(origin_index_int, 0) + 1

    candidate_rows.sort(key=lambda row: (row[4], row[5]))
    supplement_traces: list[np.ndarray] = []
    supplement_connections: list[list[int]] = []
    supplement_metrics: list[float] = []
    supplement_energy_traces: list[np.ndarray] = []
    supplement_scale_traces: list[np.ndarray] = []
    supplement_origin_indices: list[int] = []
    origin_added_counts: dict[int, int] = {}
    origin_budget_rejected = 0

    for pair, trace, energy_trace, scale_trace, metric, _endpoint_distance in candidate_rows:
        origin_index = int(pair[0])
        if candidate_mode == "remaining_origin_contacts":
            remaining_budget = max_edges_per_vertex - existing_origin_counts.get(origin_index, 0)
            if remaining_budget <= 0 or origin_added_counts.get(origin_index, 0) >= remaining_budget:
                origin_budget_rejected += 1
                continue
        elif candidate_mode == "origin_cap":
            if origin_added_counts.get(origin_index, 0) >= max_edges_per_vertex:
                origin_budget_rejected += 1
                continue

        supplement_traces.append(np.asarray(trace, dtype=np.float32))
        supplement_connections.append([int(pair[0]), int(pair[1])])
        supplement_metrics.append(float(metric))
        supplement_energy_traces.append(np.asarray(energy_trace, dtype=np.float32))
        supplement_scale_traces.append(np.asarray(scale_trace, dtype=np.int16))
        supplement_origin_indices.append(origin_index)
        origin_added_counts[origin_index] = origin_added_counts.get(origin_index, 0) + 1

    supplement_diagnostics: dict[str, Any] = {
        "watershed_join_supplement_count": len(supplement_connections),
        "watershed_per_origin_candidate_counts": {
            int(origin_index): int(count)
            for origin_index, count in origin_added_counts.items()
        },
        "watershed_origin_budget_rejected": origin_budget_rejected,
        "watershed_accepted": len(supplement_connections),
        **shared_diagnostics,
    }
    if supplement_connections:
        supplement_payload = {
            "candidate_source": "watershed",
            "traces": supplement_traces,
            "connections": np.asarray(supplement_connections, dtype=np.int32).reshape(-1, 2),
            "metrics": np.asarray(supplement_metrics, dtype=np.float32),
            "energy_traces": supplement_energy_traces,
            "scale_traces": supplement_scale_traces,
            "origin_indices": np.asarray(supplement_origin_indices, dtype=np.int32),
            "connection_sources": ["watershed"] * len(supplement_connections),
            "diagnostics": supplement_diagnostics,
        }
        _append_candidate_unit(candidates, supplement_payload)
        return candidates

    candidate_diagnostics = candidates.setdefault("diagnostics", {})
    for key, value in supplement_diagnostics.items():
        if isinstance(value, dict):
            target = candidate_diagnostics.setdefault(key, {})
            for item_key, item_value in value.items():
                target[str(item_key)] = int(target.get(str(item_key), 0)) + int(item_value)
            continue
        candidate_diagnostics[key] = int(candidate_diagnostics.get(key, 0)) + int(value)
    return candidates


def _empty_watershed_row_diagnostics() -> dict[str, int]:
    """Return the shared watershed candidate-row diagnostic counters."""
    return {
        "watershed_total_pairs": 0,
        "watershed_already_existing": 0,
        "watershed_short_trace_rejected": 0,
        "watershed_energy_rejected": 0,
        "watershed_metric_threshold_rejected": 0,
    }
