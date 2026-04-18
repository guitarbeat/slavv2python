"""Local geodesic parity salvage for edge candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .._edge_payloads import _merge_edge_diagnostics
from ..edge_primitives import (
    TraceMetadata,
    _clip_trace_indices,
    _edge_metric_from_energy_trace,
    _trace_energy_series,
    _trace_scale_series,
)
from ..vertices import _matlab_linear_indices
from .candidate_manifest import _append_candidate_unit
from .common import (
    Float32Array,
    _candidate_endpoint_pair_set,
    _candidate_incident_pair_counts,
    _vertex_center_linear_lookup,
)


def _salvage_matlab_parity_candidates_with_local_geodesics(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    microns_per_voxel: np.ndarray,
    params: dict[str, Any],
    *,
    salvage_mode: str,
    parity_metric_threshold: float | None,
) -> dict[str, Any]:
    """Recover parity candidates via bounded local geodesic searches."""
    if len(vertex_positions) < 2 or salvage_mode == "none":
        return candidates

    max_edges_per_vertex = int(params.get("number_of_edges_per_vertex", 4))
    k_nearest = max(1, int(params.get("parity_geodesic_salvage_k_nearest", 10)))
    box_margin_voxels = max(0, int(params.get("parity_geodesic_salvage_box_margin_voxels", 4)))
    max_path_ratio = float(params.get("parity_geodesic_salvage_max_path_ratio", 2.5))

    connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 2)
    frontier_origin_counts: dict[int, int] = {}
    for origin_index, count in (
        candidates.get("diagnostics", {}).get("frontier_per_origin_candidate_counts", {}).items()
    ):
        try:
            frontier_origin_counts[int(origin_index)] = int(count)
        except (TypeError, ValueError):
            continue
    frontier_terminal_rejections: dict[int, int] = {}
    for origin_index, count in (
        candidates.get("diagnostics", {}).get("frontier_per_origin_terminal_rejections", {}).items()
    ):
        try:
            rejection_count = int(count)
        except (TypeError, ValueError):
            continue
        if rejection_count > 0:
            frontier_terminal_rejections[int(origin_index)] = rejection_count

    existing_pairs = _candidate_endpoint_pair_set(connections)
    incident_pair_counts = _candidate_incident_pair_counts(connections)
    vertex_positions_microns: Float32Array = np.asarray(
        vertex_positions, dtype=np.float32
    ) * np.asarray(
        microns_per_voxel,
        dtype=np.float32,
    )
    tree = cKDTree(vertex_positions_microns)
    query_k = min(len(vertex_positions), k_nearest + 1)
    if query_k <= 1:
        return candidates

    neighbor_distances, neighbor_indices = tree.query(vertex_positions_microns, k=query_k)
    neighbor_distances = np.asarray(neighbor_distances, dtype=np.float32)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, np.newaxis]
        neighbor_distances = neighbor_distances[:, np.newaxis]

    vertex_linear_lookup = _vertex_center_linear_lookup(vertex_positions, energy.shape)
    accepted_rows: list[tuple[int, float, float, int, np.ndarray, np.ndarray, np.ndarray]] = []
    accepted_pairs: set[tuple[int, int]] = set()
    origin_added_counts: dict[int, int] = {}
    diagnostics: TraceMetadata = {
        "geodesic_join_supplement_count": 0,
        "geodesic_total_attempted_pairs": 0,
        "geodesic_existing_pair_skipped": 0,
        "geodesic_route_failed": 0,
        "geodesic_short_trace_rejected": 0,
        "geodesic_energy_rejected": 0,
        "geodesic_metric_threshold_rejected": 0,
        "geodesic_path_ratio_rejected": 0,
        "geodesic_vertex_crossing_rejected": 0,
        "geodesic_origin_budget_rejected": 0,
        "geodesic_endpoint_degree_rejected": 0,
        "geodesic_shared_neighborhood_endpoint_relaxed": 0,
        "geodesic_accepted": 0,
        "geodesic_per_origin_candidate_counts": {},
    }

    for origin_index in range(len(vertex_positions)):
        frontier_count = frontier_origin_counts.get(origin_index, 0)
        rejected_terminal_hits = frontier_terminal_rejections.get(origin_index, 0)
        shared_origin_overflow_enabled = (
            salvage_mode == "frontier_deficit_geodesic"
            and frontier_count >= max_edges_per_vertex
            and rejected_terminal_hits > 0
        )
        if salvage_mode == "frontier_deficit_geodesic" and (
            frontier_count >= max_edges_per_vertex and not shared_origin_overflow_enabled
        ):
            diagnostics["geodesic_origin_budget_rejected"] += 1
            continue

        if salvage_mode == "frontier_deficit_geodesic":
            if shared_origin_overflow_enabled:
                max_new_pairs = min(2, rejected_terminal_hits)
            else:
                max_new_pairs = max_edges_per_vertex - frontier_count
        else:
            max_new_pairs = max(1, min(max_edges_per_vertex, 2))
        if max_new_pairs <= 0:
            diagnostics["geodesic_origin_budget_rejected"] += 1
            continue

        origin_rows: list[tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for neighbor_distance, neighbor_index in zip(
            neighbor_distances[origin_index].tolist(),
            neighbor_indices[origin_index].tolist(),
        ):
            from .. import edge_candidates as edge_candidates_facade

            neighbor_index = int(neighbor_index)
            if neighbor_index < 0 or neighbor_index == origin_index:
                continue

            pair = (
                (origin_index, neighbor_index)
                if origin_index < neighbor_index
                else (neighbor_index, origin_index)
            )
            if pair in existing_pairs or pair in accepted_pairs:
                diagnostics["geodesic_existing_pair_skipped"] += 1
                continue
            relaxed_endpoint_cap = False
            blocked_by_endpoint_cap = False
            for endpoint in pair:
                if incident_pair_counts.get(endpoint, 0) < max_edges_per_vertex:
                    continue
                if endpoint == origin_index and shared_origin_overflow_enabled:
                    relaxed_endpoint_cap = True
                    continue
                blocked_by_endpoint_cap = True
                break
            if blocked_by_endpoint_cap:
                diagnostics["geodesic_endpoint_degree_rejected"] += 1
                continue
            if relaxed_endpoint_cap:
                diagnostics["geodesic_shared_neighborhood_endpoint_relaxed"] += 1

            diagnostics["geodesic_total_attempted_pairs"] += 1
            trace = edge_candidates_facade._trace_local_geodesic_between_vertices(
                energy,
                vertex_positions[origin_index],
                vertex_positions[neighbor_index],
                energy_sign,
                box_margin_voxels=box_margin_voxels,
            )
            if trace is None:
                diagnostics["geodesic_route_failed"] += 1
                continue
            if len(trace) <= 1:
                diagnostics["geodesic_short_trace_rejected"] += 1
                continue

            if len(trace) > 2:
                trace_indices = _clip_trace_indices(trace[1:-1], energy.shape)
                trace_linear = _matlab_linear_indices(trace_indices, energy.shape)
                crossed_vertex = False
                for linear_index in trace_linear.tolist():
                    vertex_index = vertex_linear_lookup.get(int(linear_index))
                    if vertex_index is None:
                        continue
                    if vertex_index not in {origin_index, neighbor_index}:
                        crossed_vertex = True
                        break
                if crossed_vertex:
                    diagnostics["geodesic_vertex_crossing_rejected"] += 1
                    continue

            energy_trace = _trace_energy_series(trace, energy)
            energy_trace_array = np.asarray(energy_trace, dtype=np.float32)
            max_energy = float(np.nanmax(energy_trace_array))
            if energy_sign < 0:
                is_invalid = max_energy >= 0
            else:
                is_invalid = float(np.nanmin(energy_trace_array)) <= 0
            if is_invalid:
                diagnostics["geodesic_energy_rejected"] += 1
                continue

            if parity_metric_threshold is not None and max_energy > parity_metric_threshold:
                diagnostics["geodesic_metric_threshold_rejected"] += 1
                continue

            straight_distance = float(
                np.linalg.norm(
                    vertex_positions_microns[origin_index]
                    - vertex_positions_microns[neighbor_index]
                )
            )
            if len(trace) > 1:
                step_vectors = np.diff(np.asarray(trace, dtype=np.float32), axis=0)
                path_length = float(
                    np.linalg.norm(
                        step_vectors * np.asarray(microns_per_voxel, dtype=np.float32), axis=1
                    ).sum()
                )
            else:
                path_length = 0.0
            if straight_distance > 0 and path_length > straight_distance * max_path_ratio:
                diagnostics["geodesic_path_ratio_rejected"] += 1
                continue

            scale_trace = _trace_scale_series(trace, scale_indices)
            metric = _edge_metric_from_energy_trace(energy_trace)
            origin_rows.append(
                (
                    metric,
                    float(neighbor_distance),
                    neighbor_index,
                    trace,
                    energy_trace,
                    scale_trace,
                )
            )

        origin_rows.sort(key=lambda row: (row[0], row[1], row[2]))
        for metric, distance, neighbor_index, trace, energy_trace, scale_trace in origin_rows:
            if origin_added_counts.get(origin_index, 0) >= max_new_pairs:
                break
            pair = (
                (origin_index, neighbor_index)
                if origin_index < neighbor_index
                else (neighbor_index, origin_index)
            )
            if pair in existing_pairs or pair in accepted_pairs:
                continue
            accepted_rows.append(
                (
                    origin_index,
                    metric,
                    float(distance),
                    neighbor_index,
                    trace,
                    energy_trace,
                    scale_trace,
                )
            )
            accepted_pairs.add(pair)
            origin_added_counts[origin_index] = origin_added_counts.get(origin_index, 0) + 1
            incident_pair_counts[pair[0]] = incident_pair_counts.get(pair[0], 0) + 1
            incident_pair_counts[pair[1]] = incident_pair_counts.get(pair[1], 0) + 1
            diagnostics["geodesic_per_origin_candidate_counts"][str(origin_index)] = int(
                origin_added_counts[origin_index]
            )

    if not accepted_rows:
        _merge_edge_diagnostics(candidates.get("diagnostics", {}), diagnostics)
        return candidates

    supplement_payload: dict[str, Any] = {
        "candidate_source": "geodesic",
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": diagnostics,
    }
    for (
        origin_index,
        metric,
        _distance,
        neighbor_index,
        trace,
        energy_trace,
        scale_trace,
    ) in accepted_rows:
        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([origin_index, neighbor_index])
        supplement_payload["metrics"].append(metric)
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(origin_index)
        supplement_payload["connection_sources"].append("geodesic")

    supplement_payload["diagnostics"]["geodesic_join_supplement_count"] = len(accepted_rows)
    supplement_payload["diagnostics"]["geodesic_accepted"] = len(accepted_rows)
    _append_candidate_unit(candidates, supplement_payload)
    return candidates
