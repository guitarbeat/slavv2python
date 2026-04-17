"""
Comparison metrics for SLAVV validation.

This module contains functions to compare vertices, edges, and network statistics
between MATLAB and Python implementations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from ._metrics.counts import (
    _count_items,
    _infer_edges_count,
    _infer_strand_count,
    _infer_vertices_count,
    _resolve_count,
)
from ._metrics.signatures import (
    _as_position_array,
    _edge_endpoint_pair_set,
    _edge_endpoint_signatures,
    _edge_signatures,
    _sample_counter_diff,
    _sample_set_diff,
    _strand_signatures,
    _vertex_signatures,
)
from ._metrics.shared_neighborhood import build_shared_neighborhood_audit  # noqa: F401


def _normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
) -> list[str]:
    """Return normalized connection-source labels when candidate provenance is available."""
    if candidate_connection_count <= 0:
        return []
    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        return []
    if len(source_values) != candidate_connection_count:
        return []
    allowed_sources = {"frontier", "watershed", "geodesic", "fallback"}
    normalized: list[str] = []
    for value in source_values:
        source_label = str(value).strip().lower()
        normalized.append(source_label if source_label in allowed_sources else "fallback")
    return normalized


def _candidate_endpoint_pair_details(
    payload: dict[str, Any],
) -> tuple[
    dict[int, set[tuple[int, int]]],
    dict[int, dict[str, set[tuple[int, int]]]],
    dict[str, set[tuple[int, int]]],
    dict[tuple[int, int], set[str]],
]:
    """Group candidate endpoint pairs by seed origin and provenance source."""
    connections = np.asarray(payload.get("connections", np.array([])))
    if connections.size == 0:
        return (
            {},
            {},
            {"frontier": set(), "watershed": set(), "geodesic": set(), "fallback": set()},
            {},
        )
    if connections.ndim == 1:
        connections = connections.reshape(1, -1)

    origins = np.asarray(payload.get("origin_indices", np.array([])), dtype=np.int32).reshape(-1)
    if origins.size != len(connections):
        origins = connections[:, 0].astype(np.int32, copy=False)
    connection_sources = _normalize_candidate_connection_sources(
        payload.get("connection_sources"),
        len(connections),
    )

    pairs_by_seed_origin: dict[int, set[tuple[int, int]]] = {}
    source_pairs_by_seed_origin: dict[int, dict[str, set[tuple[int, int]]]] = {}
    pairs_by_source: dict[str, set[tuple[int, int]]] = {
        "frontier": set(),
        "watershed": set(),
        "geodesic": set(),
        "fallback": set(),
    }
    pair_sources: dict[tuple[int, int], set[str]] = {}

    for index, connection in enumerate(connections):
        pair = [int(value) for value in np.asarray(connection).tolist()[:2]]
        if len(pair) < 2 or pair[0] < 0 or pair[1] < 0:
            continue
        endpoint_pair = tuple(sorted(pair))
        origin = int(origins[index]) if index < len(origins) else int(endpoint_pair[0])
        pairs_by_seed_origin.setdefault(origin, set()).add(endpoint_pair)
        if index >= len(connection_sources):
            continue
        source_label = connection_sources[index]
        pairs_by_source[source_label].add(endpoint_pair)
        pair_sources.setdefault(endpoint_pair, set()).add(source_label)
        source_pairs_by_seed_origin.setdefault(origin, {}).setdefault(source_label, set()).add(
            endpoint_pair
        )

    return pairs_by_seed_origin, source_pairs_by_seed_origin, pairs_by_source, pair_sources


def _coerce_str_int_map(raw: Any) -> dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    converted: dict[int, int] = {}
    for key, value in raw.items():
        try:
            converted[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return converted


def _candidate_audit_summary(candidate_audit: dict[str, Any] | None) -> dict[str, Any] | None:
    """Create a compact summary payload from a candidate-audit artifact."""
    if not isinstance(candidate_audit, dict):
        return None

    per_origin = candidate_audit.get("per_origin_summary")
    top_per_origin = []
    if isinstance(per_origin, list):
        top_per_origin = sorted(
            [item for item in per_origin if isinstance(item, dict)],
            key=lambda item: (
                -int(item.get("watershed_candidate_count", 0)),
                -int(item.get("geodesic_candidate_count", 0)),
                -int(item.get("frontier_candidate_count", 0)),
                -int(item.get("candidate_connection_count", 0)),
                int(item.get("origin_index", 0)),
            ),
        )[:10]

    return {
        "schema_version": int(candidate_audit.get("schema_version", 1)),
        "vertex_count": int(candidate_audit.get("vertex_count", 0)),
        "use_frontier_tracer": bool(candidate_audit.get("use_frontier_tracer", False)),
        "candidate_traces": int(candidate_audit.get("candidate_traces", 0)),
        "candidate_connection_count": int(candidate_audit.get("candidate_connection_count", 0)),
        "candidate_origin_count": int(candidate_audit.get("candidate_origin_count", 0)),
        "source_breakdown": candidate_audit.get("source_breakdown", {}),
        "frontier_per_origin_candidate_counts": _coerce_str_int_map(
            candidate_audit.get("frontier_per_origin_candidate_counts")
        ),
        "watershed_per_origin_candidate_counts": _coerce_str_int_map(
            candidate_audit.get("watershed_per_origin_candidate_counts")
        ),
        "geodesic_per_origin_candidate_counts": _coerce_str_int_map(
            candidate_audit.get("geodesic_per_origin_candidate_counts")
        ),
        "pair_source_breakdown": candidate_audit.get("pair_source_breakdown", {}),
        "top_origin_summaries": top_per_origin,
        "diagnostic_counters": candidate_audit.get("diagnostic_counters", {}),
    }


def _edge_endpoint_pairs_by_seed_origin(
    payload: dict[str, Any],
) -> dict[int, set[tuple[int, int]]]:
    """Group unique endpoint pairs by the recorded seed origin for candidate edges."""
    grouped, _, _, _ = _candidate_endpoint_pair_details(payload)
    return grouped


def _candidate_endpoint_pairs_by_source(
    payload: dict[str, Any],
) -> dict[str, set[tuple[int, int]]]:
    """Group unique candidate endpoint pairs by their recorded source label."""
    _, _, grouped, _ = _candidate_endpoint_pair_details(payload)
    return {label: pairs for label, pairs in grouped.items() if pairs}


def _chosen_candidate_source_summary(
    python_edges: dict[str, Any],
    candidate_edges: dict[str, Any],
    matlab_endpoint_pairs: set[tuple[int, int]],
) -> dict[str, Any] | None:
    """Summarize which candidate sources survived into the final chosen Python edges."""
    candidate_connections = np.asarray(candidate_edges.get("connections", np.array([])))
    if candidate_connections.size == 0:
        return None
    if candidate_connections.ndim == 1:
        candidate_connections = candidate_connections.reshape(1, -1)

    chosen_indices = np.asarray(
        python_edges.get("chosen_candidate_indices", np.array([], dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if chosen_indices.size == 0:
        return None

    connection_sources = _normalize_candidate_connection_sources(
        candidate_edges.get("connection_sources"),
        len(candidate_connections),
    )
    if not connection_sources:
        return None

    allowed_sources = ("frontier", "watershed", "geodesic", "fallback")
    source_counts = dict.fromkeys(allowed_sources, 0)
    chosen_watershed_pairs: set[tuple[int, int]] = set()
    chosen_geodesic_pairs: set[tuple[int, int]] = set()
    python_connections = np.asarray(python_edges.get("connections", np.array([])))
    if python_connections.size == 0:
        python_connections = np.empty((0, 2), dtype=np.int32)
    elif python_connections.ndim == 1:
        python_connections = python_connections.reshape(1, -1)
    python_energies = np.asarray(
        python_edges.get("energies", np.array([], dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    python_trace_lengths = np.array(
        [len(np.asarray(trace)) for trace in python_edges.get("traces", [])],
        dtype=np.int32,
    )
    source_breakdown: dict[str, dict[str, Any]] = {}

    for raw_index in chosen_indices.tolist():
        candidate_index = int(raw_index)
        if candidate_index < 0 or candidate_index >= len(candidate_connections):
            continue
        source_label = connection_sources[candidate_index]
        if source_label not in source_counts:
            continue
        source_counts[source_label] += 1
        pair = [
            int(value) for value in np.asarray(candidate_connections[candidate_index]).tolist()[:2]
        ]
        if len(pair) < 2 or pair[0] < 0 or pair[1] < 0:
            continue
        endpoint_pair = tuple(sorted(pair))
        if source_label == "watershed":
            chosen_watershed_pairs.add(endpoint_pair)
        elif source_label == "geodesic":
            chosen_geodesic_pairs.add(endpoint_pair)

    if len(python_connections) > 0:
        python_endpoint_pairs = [
            tuple(sorted(int(value) for value in np.asarray(connection).tolist()[:2]))
            for connection in python_connections
        ]
        python_connection_sources = _normalize_candidate_connection_sources(
            python_edges.get("connection_sources"),
            len(python_connections),
        )
        if not python_connection_sources and len(chosen_indices) == len(python_connections):
            python_connection_sources = []
            for raw_index in chosen_indices.tolist():
                candidate_index = int(raw_index)
                if 0 <= candidate_index < len(connection_sources):
                    python_connection_sources.append(connection_sources[candidate_index])
                else:
                    python_connection_sources.append("fallback")

        if len(python_connection_sources) == len(python_connections):
            for source_label in allowed_sources:
                source_mask = np.array(
                    [label == source_label for label in python_connection_sources],
                    dtype=bool,
                )
                if not np.any(source_mask):
                    continue

                source_pairs = [
                    pair
                    for pair, include in zip(python_endpoint_pairs, source_mask.tolist())
                    if include
                ]
                matched_mask = np.array(
                    [pair in matlab_endpoint_pairs for pair in source_pairs],
                    dtype=bool,
                )
                source_energies = (
                    python_energies[source_mask]
                    if python_energies.size == len(python_connections)
                    else np.empty((0,), dtype=np.float32)
                )
                source_lengths = (
                    python_trace_lengths[source_mask]
                    if python_trace_lengths.size == len(python_connections)
                    else np.empty((0,), dtype=np.int32)
                )

                summary: dict[str, Any] = {
                    "chosen_edge_count": int(np.sum(source_mask)),
                    "matched_matlab_edge_count": int(np.sum(matched_mask)),
                    "extra_python_edge_count": int(np.sum(~matched_mask)),
                }

                for label, submask in (
                    ("all", np.ones(np.sum(source_mask), dtype=bool)),
                    ("matched", matched_mask),
                    ("extra", ~matched_mask),
                ):
                    if not np.any(submask):
                        continue
                    stats: dict[str, Any] = {"edge_count": int(np.sum(submask))}
                    if source_energies.size == len(source_pairs):
                        stats["median_energy"] = float(np.median(source_energies[submask]))
                    if source_lengths.size == len(source_pairs):
                        stats["median_length"] = float(np.median(source_lengths[submask]))
                    summary[label] = stats
                source_breakdown[source_label] = summary

    chosen_watershed_matched = len(chosen_watershed_pairs & matlab_endpoint_pairs)
    result = {
        "counts": source_counts,
        "watershed_endpoint_pair_count": len(chosen_watershed_pairs),
        "watershed_matched_matlab_endpoint_pair_count": chosen_watershed_matched,
        "watershed_extra_python_endpoint_pair_count": len(chosen_watershed_pairs)
        - chosen_watershed_matched,
        "geodesic_endpoint_pair_count": len(chosen_geodesic_pairs),
        "geodesic_matched_matlab_endpoint_pair_count": len(
            chosen_geodesic_pairs & matlab_endpoint_pairs
        ),
        "geodesic_extra_python_endpoint_pair_count": len(chosen_geodesic_pairs)
        - len(chosen_geodesic_pairs & matlab_endpoint_pairs),
    }
    if source_breakdown:
        result["source_breakdown"] = source_breakdown
    return result


def _frontier_missing_vertex_overlap_summary(
    python_edges: dict[str, Any],
    candidate_edges: dict[str, Any],
    matlab_endpoint_pairs: set[tuple[int, int]],
) -> dict[str, Any] | None:
    """Summarize whether extra frontier edges cluster around missing MATLAB vertices."""
    python_connections = np.asarray(python_edges.get("connections", np.array([])))
    if python_connections.size == 0:
        return None
    if python_connections.ndim == 1:
        python_connections = python_connections.reshape(1, -1)
    connection_sources = _normalize_candidate_connection_sources(
        python_edges.get("connection_sources"),
        len(python_connections),
    )
    if len(connection_sources) != len(python_connections):
        return None

    python_endpoint_pairs = [
        tuple(sorted(int(value) for value in np.asarray(connection).tolist()[:2]))
        for connection in python_connections
    ]
    python_endpoint_pair_set = set(python_endpoint_pairs)
    missing_matlab_pairs = matlab_endpoint_pairs - python_endpoint_pair_set
    if not missing_matlab_pairs:
        return None
    missing_vertices = {vertex for pair in missing_matlab_pairs for vertex in pair}

    python_energies = np.asarray(
        python_edges.get("energies", np.array([], dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    python_trace_lengths = np.array(
        [len(np.asarray(trace)) for trace in python_edges.get("traces", [])],
        dtype=np.int32,
    )

    extra_frontier_entries = []
    for index, (pair, source_label) in enumerate(zip(python_endpoint_pairs, connection_sources)):
        if source_label != "frontier" or pair in matlab_endpoint_pairs:
            continue
        entry: dict[str, Any] = {
            "pair": pair,
            "shares_missing_vertex": any(vertex in missing_vertices for vertex in pair),
        }
        if index < len(python_energies):
            entry["energy"] = float(python_energies[index])
        if index < len(python_trace_lengths):
            entry["trace_length"] = int(python_trace_lengths[index])
        extra_frontier_entries.append(entry)

    if not extra_frontier_entries:
        return None

    candidate_pair_sets = _candidate_endpoint_pairs_by_source(candidate_edges)
    candidate_endpoint_pair_set = set()
    for pairs in candidate_pair_sets.values():
        candidate_endpoint_pair_set.update(pairs)
    candidate_incident_by_vertex = _incident_endpoint_pairs_by_vertex(candidate_endpoint_pair_set)
    frontier_candidate_incident_by_vertex = _incident_endpoint_pairs_by_vertex(
        candidate_pair_sets.get("frontier", set())
    )
    watershed_candidate_incident_by_vertex = _incident_endpoint_pairs_by_vertex(
        candidate_pair_sets.get("watershed", set())
    )
    geodesic_candidate_incident_by_vertex = _incident_endpoint_pairs_by_vertex(
        candidate_pair_sets.get("geodesic", set())
    )

    chosen_pair_sets = {"frontier": set(), "watershed": set(), "geodesic": set(), "fallback": set()}
    for pair, source_label in zip(python_endpoint_pairs, connection_sources):
        if source_label in chosen_pair_sets:
            chosen_pair_sets[source_label].add(pair)
    chosen_incident_by_vertex = _incident_endpoint_pairs_by_vertex(python_endpoint_pair_set)
    frontier_chosen_incident_by_vertex = _incident_endpoint_pairs_by_vertex(
        chosen_pair_sets["frontier"]
    )
    watershed_chosen_incident_by_vertex = _incident_endpoint_pairs_by_vertex(
        chosen_pair_sets["watershed"]
    )
    geodesic_chosen_incident_by_vertex = _incident_endpoint_pairs_by_vertex(
        chosen_pair_sets["geodesic"]
    )

    extra_frontier_entries.sort(key=lambda item: item.get("energy", np.inf))
    top_overlap_counts: dict[str, Any] = {}
    for threshold in (20, 50, 100):
        if subset := extra_frontier_entries[:threshold]:
            top_overlap_counts[str(threshold)] = {
                "threshold": threshold,
                "shared_missing_vertex_count": int(
                    sum(bool(item["shares_missing_vertex"]) for item in subset)
                ),
                "evaluated_edge_count": len(subset),
            }

    missing_by_vertex = _incident_endpoint_pairs_by_vertex(missing_matlab_pairs)
    extra_frontier_pairs = [entry["pair"] for entry in extra_frontier_entries]
    extra_frontier_by_vertex = _incident_endpoint_pairs_by_vertex(set(extra_frontier_pairs))
    shared_vertices = sorted(set(missing_by_vertex) & set(extra_frontier_by_vertex))
    top_shared_vertices = [
        {
            "vertex_index": int(vertex),
            "missing_matlab_endpoint_pair_count": len(missing_by_vertex[vertex]),
            "extra_frontier_endpoint_pair_count": len(extra_frontier_by_vertex[vertex]),
            "missing_matlab_pairs_present_in_candidates": len(
                missing_by_vertex[vertex] & candidate_incident_by_vertex.get(vertex, set())
            ),
            "missing_matlab_pairs_present_in_frontier_candidates": len(
                missing_by_vertex[vertex] & frontier_candidate_incident_by_vertex.get(vertex, set())
            ),
            "missing_matlab_pairs_present_in_watershed_candidates": len(
                missing_by_vertex[vertex]
                & watershed_candidate_incident_by_vertex.get(vertex, set())
            ),
            "missing_matlab_pairs_present_in_geodesic_candidates": len(
                missing_by_vertex[vertex] & geodesic_candidate_incident_by_vertex.get(vertex, set())
            ),
            "candidate_incident_endpoint_pair_count": len(
                candidate_incident_by_vertex.get(vertex, set())
            ),
            "frontier_candidate_incident_endpoint_pair_count": len(
                frontier_candidate_incident_by_vertex.get(vertex, set())
            ),
            "watershed_candidate_incident_endpoint_pair_count": len(
                watershed_candidate_incident_by_vertex.get(vertex, set())
            ),
            "geodesic_candidate_incident_endpoint_pair_count": len(
                geodesic_candidate_incident_by_vertex.get(vertex, set())
            ),
            "chosen_incident_endpoint_pair_count": len(
                chosen_incident_by_vertex.get(vertex, set())
            ),
            "chosen_frontier_incident_endpoint_pair_count": len(
                frontier_chosen_incident_by_vertex.get(vertex, set())
            ),
            "chosen_watershed_incident_endpoint_pair_count": len(
                watershed_chosen_incident_by_vertex.get(vertex, set())
            ),
            "chosen_geodesic_incident_endpoint_pair_count": len(
                geodesic_chosen_incident_by_vertex.get(vertex, set())
            ),
            "missing_matlab_endpoint_pair_samples": sorted(missing_by_vertex[vertex])[:3],
            "extra_frontier_endpoint_pair_samples": sorted(extra_frontier_by_vertex[vertex])[:3],
        }
        for vertex in shared_vertices
    ]
    top_shared_vertices.sort(
        key=lambda item: (
            -int(item["missing_matlab_endpoint_pair_count"]),
            -int(item["extra_frontier_endpoint_pair_count"]),
            int(item["vertex_index"]),
        )
    )

    return {
        "extra_frontier_edge_count": len(extra_frontier_entries),
        "shared_missing_vertex_edge_count": int(
            sum(bool(entry["shares_missing_vertex"]) for entry in extra_frontier_entries)
        ),
        "missing_matlab_pair_count": len(missing_matlab_pairs),
        "shared_vertex_count": len(shared_vertices),
        "top_strength_overlap_counts": top_overlap_counts,
        "top_shared_vertices": top_shared_vertices[:10],
        "strongest_extra_frontier_samples": extra_frontier_entries[:10],
    }


def _incident_endpoint_pairs_by_vertex(
    endpoint_pairs: set[tuple[int, int]],
) -> dict[int, set[tuple[int, int]]]:
    """Group endpoint pairs by each incident vertex."""
    grouped: dict[int, set[tuple[int, int]]] = {}
    for pair in endpoint_pairs:
        start_vertex, end_vertex = (int(value) for value in pair)
        grouped.setdefault(start_vertex, set()).add(pair)
        grouped.setdefault(end_vertex, set()).add(pair)
    return grouped


def _missing_matlab_seed_origin_samples(
    matlab_endpoint_pairs: set[tuple[int, int]],
    missing_pairs_by_vertex: dict[int, set[tuple[int, int]]],
    candidate_edges: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Summarize missing MATLAB endpoint pairs by candidate seed origin."""
    if not missing_pairs_by_vertex:
        return []

    matlab_pairs_by_vertex = _incident_endpoint_pairs_by_vertex(matlab_endpoint_pairs)
    candidate_pairs_by_seed_origin, source_pairs_by_seed_origin, _, _ = (
        _candidate_endpoint_pair_details(candidate_edges)
    )

    samples = []
    for seed_origin_index in sorted(missing_pairs_by_vertex):
        missing_pairs = missing_pairs_by_vertex[seed_origin_index]
        matlab_incident_pairs = matlab_pairs_by_vertex.get(seed_origin_index, set())
        candidate_pairs = candidate_pairs_by_seed_origin.get(seed_origin_index, set())
        source_pairs = source_pairs_by_seed_origin.get(seed_origin_index, {})
        matched_candidate_pairs = candidate_pairs & matlab_incident_pairs
        extra_candidate_pairs = candidate_pairs - matlab_incident_pairs
        samples.append(
            {
                "seed_origin_index": int(seed_origin_index),
                "matlab_incident_endpoint_pair_count": len(matlab_incident_pairs),
                "missing_matlab_incident_endpoint_pair_count": len(missing_pairs),
                "matched_matlab_incident_endpoint_pair_count": len(matched_candidate_pairs),
                "candidate_endpoint_pair_count": len(candidate_pairs),
                "extra_candidate_endpoint_pair_count": len(extra_candidate_pairs),
                "frontier_candidate_endpoint_pair_count": len(source_pairs.get("frontier", set())),
                "watershed_candidate_endpoint_pair_count": len(
                    source_pairs.get("watershed", set())
                ),
                "fallback_candidate_endpoint_pair_count": len(source_pairs.get("fallback", set())),
                "missing_matlab_incident_endpoint_pair_samples": sorted(missing_pairs)[:3],
                "candidate_endpoint_pair_samples": sorted(candidate_pairs)[:3],
                "extra_candidate_endpoint_pair_samples": sorted(extra_candidate_pairs)[:3],
            }
        )

    samples.sort(
        key=lambda item: (
            -int(item["missing_matlab_incident_endpoint_pair_count"]),
            int(item["candidate_endpoint_pair_count"]),
            int(item["seed_origin_index"]),
        )
    )
    return samples[:limit]


def _extra_candidate_seed_origin_samples(
    matlab_endpoint_pairs: set[tuple[int, int]],
    candidate_edges: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Summarize extra candidate endpoint pairs by the recorded seed origin."""
    candidate_pairs_by_seed_origin, source_pairs_by_seed_origin, _, _ = (
        _candidate_endpoint_pair_details(candidate_edges)
    )
    if not candidate_pairs_by_seed_origin:
        return []

    samples = []
    for seed_origin_index, candidate_pairs in candidate_pairs_by_seed_origin.items():
        extra_candidate_pairs = candidate_pairs - matlab_endpoint_pairs
        if not extra_candidate_pairs:
            continue
        source_pairs = source_pairs_by_seed_origin.get(seed_origin_index, {})
        samples.append(
            {
                "seed_origin_index": int(seed_origin_index),
                "candidate_endpoint_pair_count": len(candidate_pairs),
                "extra_candidate_endpoint_pair_count": len(extra_candidate_pairs),
                "frontier_candidate_endpoint_pair_count": len(source_pairs.get("frontier", set())),
                "watershed_candidate_endpoint_pair_count": len(
                    source_pairs.get("watershed", set())
                ),
                "fallback_candidate_endpoint_pair_count": len(source_pairs.get("fallback", set())),
                "candidate_endpoint_pair_samples": sorted(candidate_pairs)[:3],
                "extra_candidate_endpoint_pair_samples": sorted(extra_candidate_pairs)[:3],
            }
        )

    samples.sort(
        key=lambda item: (
            -int(item["extra_candidate_endpoint_pair_count"]),
            -int(item["candidate_endpoint_pair_count"]),
            int(item["seed_origin_index"]),
        )
    )
    return samples[:limit]


def match_vertices(
    matlab_positions: np.ndarray, python_positions: np.ndarray, distance_threshold: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match vertices between MATLAB and Python using one-to-one nearest neighbors."""
    matlab_positions = _as_position_array(matlab_positions)
    python_positions = _as_position_array(python_positions)

    if matlab_positions.size == 0 or python_positions.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Use only spatial coordinates (first 3 columns)
    matlab_xyz = matlab_positions[:, :3]
    python_xyz = python_positions[:, :3]

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(python_xyz)

    # Find nearest Python vertex for each MATLAB vertex
    distances, python_indices = tree.query(matlab_xyz)

    candidate_pairs = [
        (float(distance), int(matlab_index), int(python_index))
        for matlab_index, (distance, python_index) in enumerate(zip(distances, python_indices))
        if distance < distance_threshold
    ]
    candidate_pairs.sort()

    matched_matlab = []
    matched_python = []
    matched_distances = []
    used_matlab: set[int] = set()
    used_python: set[int] = set()
    for distance, matlab_index, python_index in candidate_pairs:
        if matlab_index in used_matlab or python_index in used_python:
            continue
        used_matlab.add(matlab_index)
        used_python.add(python_index)
        matched_matlab.append(matlab_index)
        matched_python.append(python_index)
        matched_distances.append(distance)

    return (
        np.asarray(matched_matlab, dtype=np.int32),
        np.asarray(matched_python, dtype=np.int32),
        np.asarray(matched_distances, dtype=float),
    )


def compare_vertices(matlab_verts: dict[str, Any], python_verts: dict[str, Any]) -> dict[str, Any]:
    """Compare vertex information between MATLAB and Python."""
    comparison = {
        "matlab_count": _infer_vertices_count(matlab_verts),
        "python_count": _infer_vertices_count(python_verts),
        "count_difference": 0,
        "count_percent_difference": 0.0,
        "position_rmse": None,
        "matched_vertices": 0,
        "unmatched_matlab": 0,
        "unmatched_python": 0,
        "radius_correlation": None,
        "radius_stats": {},
        "exact_positions_scales_match": False,
        "exact_positions_scales_energies_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
    }

    matlab_count = comparison["matlab_count"]
    python_count = comparison["python_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["count_percent_difference"] = (
                comparison["count_difference"] / avg_count
            ) * 100.0

    # Match vertices if both have data
    matlab_positions = _as_position_array(matlab_verts.get("positions", np.array([])))
    python_positions = _as_position_array(python_verts.get("positions", np.array([])))

    if matlab_positions.size > 0 and python_positions.size > 0:
        matlab_idx, python_idx, distances = match_vertices(matlab_positions, python_positions)
        unique_python_idx = np.unique(python_idx)

        comparison["matched_vertices"] = len(matlab_idx)
        comparison["unmatched_matlab"] = matlab_count - len(matlab_idx)
        comparison["unmatched_python"] = python_count - len(unique_python_idx)

        if len(distances) > 0:
            comparison["position_rmse"] = float(np.sqrt(np.mean(distances**2)))
            comparison["position_mean_distance"] = float(np.mean(distances))
            comparison["position_median_distance"] = float(np.median(distances))
            comparison["position_95th_percentile"] = float(np.percentile(distances, 95))

        # Compare radii for matched vertices
        matlab_radii = np.asarray(matlab_verts.get("radii", np.array([])))
        python_radii = np.asarray(python_verts.get("radii", np.array([])))

        if (
            len(matlab_idx) > 0
            and len(unique_python_idx) == len(matlab_idx)
            and matlab_radii.size > 0
            and python_radii.size > 0
        ):
            matched_matlab_radii = matlab_radii[matlab_idx]
            matched_python_radii = python_radii[python_idx]

            # Compute correlation
            if len(matched_matlab_radii) > 1:
                pearson_r, pearson_p = stats.pearsonr(matched_matlab_radii, matched_python_radii)
                spearman_r, spearman_p = stats.spearmanr(matched_matlab_radii, matched_python_radii)

                comparison["radius_correlation"] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }

            # Radius statistics
            comparison["radius_stats"] = {
                "matlab_mean": float(np.mean(matched_matlab_radii)),
                "matlab_std": float(np.std(matched_matlab_radii)),
                "python_mean": float(np.mean(matched_python_radii)),
                "python_std": float(np.std(matched_python_radii)),
                "mean_difference": float(np.mean(matched_matlab_radii - matched_python_radii)),
                "rmse": float(np.sqrt(np.mean((matched_matlab_radii - matched_python_radii) ** 2))),
            }

    coords_scales_matlab, coords_scales_energy_matlab = _vertex_signatures(matlab_verts)
    coords_scales_python, coords_scales_energy_python = _vertex_signatures(python_verts)
    coords_scales_counter_matlab = Counter(coords_scales_matlab)
    coords_scales_counter_python = Counter(coords_scales_python)
    coords_scales_energy_counter_matlab = Counter(coords_scales_energy_matlab)
    coords_scales_energy_counter_python = Counter(coords_scales_energy_python)

    comparison["exact_positions_scales_match"] = (
        coords_scales_counter_matlab == coords_scales_counter_python
    )
    comparison["exact_positions_scales_energies_match"] = (
        coords_scales_energy_counter_matlab == coords_scales_energy_counter_python
    )
    comparison["matlab_only_samples"] = _sample_counter_diff(
        coords_scales_counter_matlab, coords_scales_counter_python
    )
    comparison["python_only_samples"] = _sample_counter_diff(
        coords_scales_counter_python, coords_scales_counter_matlab
    )

    return comparison


def compare_edges(
    matlab_edges: dict[str, Any],
    python_edges: dict[str, Any],
    candidate_edges: dict[str, Any] | None = None,
    candidate_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare edge information between MATLAB and Python."""
    comparison = {
        "matlab_count": _infer_edges_count(matlab_edges),
        "python_count": _infer_edges_count(python_edges),
        "count_difference": 0,
        "count_percent_difference": 0.0,
        "total_length": {},
        "exact_match": False,
        "exact_endpoint_pairs_match": False,
        "exact_trace_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
        "endpoint_pair_matlab_only_samples": [],
        "endpoint_pair_python_only_samples": [],
        "diagnostics": {
            "matlab": matlab_edges.get("diagnostics", {}),
            "python": python_edges.get("diagnostics", {}),
        },
    }

    matlab_count = comparison["matlab_count"]
    python_count = comparison["python_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["count_percent_difference"] = (
                comparison["count_difference"] / avg_count
            ) * 100.0

    # Compare total lengths if available
    matlab_total_length = matlab_edges.get("total_length", 0.0)
    if matlab_total_length > 0:
        comparison["total_length"]["matlab"] = float(matlab_total_length)

    # Calculate Python edge lengths
    python_traces = python_edges.get("traces", [])
    if _count_items(python_traces) > 0:
        python_total_length = 0.0
        for trace in python_traces:
            trace_array = np.asarray(trace)
            if trace_array.size > 0 and trace_array.ndim == 2 and trace_array.shape[0] > 1:
                diffs = np.diff(trace_array[:, :3], axis=0)
                lengths = np.sqrt(np.sum(diffs**2, axis=1))
                python_total_length += np.sum(lengths)

        comparison["total_length"]["python"] = float(python_total_length)

        if matlab_total_length > 0 and python_total_length > 0:
            comparison["total_length"]["difference"] = float(
                abs(matlab_total_length - python_total_length)
            )
            comparison["total_length"]["percent_difference"] = float(
                (
                    comparison["total_length"]["difference"]
                    / ((matlab_total_length + python_total_length) / 2.0)
                )
                * 100.0
            )

    include_trace = (
        _count_items(matlab_edges.get("traces")) > 0
        and _count_items(python_edges.get("traces")) > 0
    )
    include_energy = (
        _count_items(matlab_edges.get("energies")) > 0
        and _count_items(python_edges.get("energies")) > 0
    )
    matlab_counter = Counter(_edge_signatures(matlab_edges, include_trace, include_energy))
    python_counter = Counter(_edge_signatures(python_edges, include_trace, include_energy))
    comparison["exact_match"] = matlab_counter == python_counter
    comparison["exact_trace_match"] = comparison["exact_match"] if include_trace else False
    comparison["matlab_only_samples"] = _sample_counter_diff(matlab_counter, python_counter)
    comparison["python_only_samples"] = _sample_counter_diff(python_counter, matlab_counter)

    matlab_endpoint_counter = Counter(_edge_endpoint_signatures(matlab_edges))
    python_endpoint_counter = Counter(_edge_endpoint_signatures(python_edges))
    matlab_endpoint_pairs = set(matlab_endpoint_counter)
    python_endpoint_pairs = set(python_endpoint_counter)
    comparison["exact_endpoint_pairs_match"] = matlab_endpoint_counter == python_endpoint_counter
    comparison["matched_endpoint_pair_count"] = len(matlab_endpoint_pairs & python_endpoint_pairs)
    comparison["missing_endpoint_pair_count"] = len(matlab_endpoint_pairs - python_endpoint_pairs)
    comparison["extra_endpoint_pair_count"] = len(python_endpoint_pairs - matlab_endpoint_pairs)
    comparison["endpoint_pair_matlab_only_samples"] = _sample_counter_diff(
        matlab_endpoint_counter, python_endpoint_counter
    )
    comparison["endpoint_pair_python_only_samples"] = _sample_counter_diff(
        python_endpoint_counter, matlab_endpoint_counter
    )
    if candidate_edges is not None:
        candidate_endpoint_pairs = _edge_endpoint_pair_set(candidate_edges)
        missing_matlab_pairs = matlab_endpoint_pairs - candidate_endpoint_pairs
        extra_candidate_pairs = candidate_endpoint_pairs - matlab_endpoint_pairs
        missing_pairs_by_vertex = _incident_endpoint_pairs_by_vertex(missing_matlab_pairs)
        _, _, _, pair_sources = _candidate_endpoint_pair_details(candidate_edges)

        # Split candidate counts by source (frontier vs watershed supplement)
        candidate_diag = candidate_edges.get("diagnostics", {})
        if source_pairs := _candidate_endpoint_pairs_by_source(candidate_edges):
            frontier_count = len(source_pairs.get("frontier", set()))
            supplement_count = len(source_pairs.get("watershed", set()))
            fallback_count = len(source_pairs.get("fallback", set()))
            frontier_pair_samples = sorted(source_pairs.get("frontier", set()))[:3]
            supplement_pair_samples = sorted(source_pairs.get("watershed", set()))[:3]
            fallback_pair_samples = sorted(source_pairs.get("fallback", set()))[:3]
        elif candidate_audit is not None:
            source_breakdown = candidate_audit.get("source_breakdown", {})
            frontier_count = int(
                source_breakdown.get("frontier", {}).get("candidate_endpoint_pair_count", 0)
            )
            supplement_count = int(
                source_breakdown.get("watershed", {}).get("candidate_endpoint_pair_count", 0)
            )
            fallback_count = int(
                source_breakdown.get("fallback", {}).get("candidate_endpoint_pair_count", 0)
            )
            frontier_pair_samples = list(
                source_breakdown.get("frontier", {}).get("candidate_endpoint_pair_samples", [])
            )[:3]
            supplement_pair_samples = list(
                source_breakdown.get("watershed", {}).get("candidate_endpoint_pair_samples", [])
            )[:3]
            fallback_pair_samples = list(
                source_breakdown.get("fallback", {}).get("candidate_endpoint_pair_samples", [])
            )[:3]
        else:
            supplement_count = int(candidate_diag.get("watershed_join_supplement_count", 0))
            frontier_count = max(0, len(candidate_endpoint_pairs) - supplement_count)
            fallback_count = 0
            frontier_pair_samples = []
            supplement_pair_samples = []
            fallback_pair_samples = []
        frontier_only_pairs = sorted(
            pair for pair, sources in pair_sources.items() if sources == {"frontier"}
        )
        watershed_only_pairs = sorted(
            pair for pair, sources in pair_sources.items() if sources == {"watershed"}
        )
        fallback_only_pairs = sorted(
            pair for pair, sources in pair_sources.items() if sources == {"fallback"}
        )
        multi_source_pairs = sorted(
            pair for pair, sources in pair_sources.items() if len(sources) > 1
        )

        coverage: dict[str, Any] = {
            "candidate_endpoint_pair_count": len(candidate_endpoint_pairs),
            "frontier_candidate_endpoint_pair_count": frontier_count,
            "supplement_candidate_endpoint_pair_count": supplement_count,
            "fallback_candidate_endpoint_pair_count": fallback_count,
            "matlab_endpoint_pair_count": len(matlab_endpoint_pairs),
            "python_endpoint_pair_count": len(python_endpoint_pairs),
            "matched_matlab_endpoint_pair_count": len(
                matlab_endpoint_pairs & candidate_endpoint_pairs
            ),
            "missing_matlab_endpoint_pair_count": len(missing_matlab_pairs),
            "extra_candidate_endpoint_pair_count": len(extra_candidate_pairs),
            "matlab_pairs_fully_covered": not missing_matlab_pairs,
            "missing_matlab_endpoint_pair_samples": _sample_set_diff(
                matlab_endpoint_pairs, candidate_endpoint_pairs
            ),
            "extra_candidate_endpoint_pair_samples": _sample_set_diff(
                candidate_endpoint_pairs, matlab_endpoint_pairs
            ),
            "missing_matlab_seed_origin_count": len(missing_pairs_by_vertex),
            "missing_matlab_seed_origin_samples": _missing_matlab_seed_origin_samples(
                matlab_endpoint_pairs,
                missing_pairs_by_vertex,
                candidate_edges,
            ),
            "extra_candidate_seed_origin_samples": _extra_candidate_seed_origin_samples(
                matlab_endpoint_pairs,
                candidate_edges,
            ),
            "frontier_candidate_endpoint_pair_samples": frontier_pair_samples,
            "supplement_candidate_endpoint_pair_samples": supplement_pair_samples,
            "fallback_candidate_endpoint_pair_samples": fallback_pair_samples,
            "frontier_only_candidate_endpoint_pair_count": len(frontier_only_pairs),
            "watershed_only_candidate_endpoint_pair_count": len(watershed_only_pairs),
            "fallback_only_candidate_endpoint_pair_count": len(fallback_only_pairs),
            "multi_source_candidate_endpoint_pair_count": len(multi_source_pairs),
            "watershed_only_candidate_endpoint_pair_samples": watershed_only_pairs[:3],
        }
        coverage["extra_candidate_seed_origin_count"] = len(
            coverage["extra_candidate_seed_origin_samples"]
        )
        # Propagate watershed rejection breakdowns when available
        for diag_key in (
            "watershed_total_pairs",
            "watershed_already_existing",
            "watershed_short_trace_rejected",
            "watershed_energy_rejected",
            "watershed_reachability_rejected",
            "watershed_mutual_frontier_rejected",
            "watershed_endpoint_degree_rejected",
            "watershed_cap_rejected",
            "watershed_accepted",
            "frontier_origins_with_candidates",
            "frontier_origins_without_candidates",
        ):
            if diag_key in candidate_diag:
                coverage[diag_key] = int(candidate_diag[diag_key])
        comparison["diagnostics"]["candidate_endpoint_coverage"] = coverage
        chosen_source_summary = _chosen_candidate_source_summary(
            python_edges,
            candidate_edges,
            matlab_endpoint_pairs,
        )
        if chosen_source_summary is not None:
            comparison["diagnostics"]["chosen_candidate_sources"] = chosen_source_summary
        frontier_overlap_summary = _frontier_missing_vertex_overlap_summary(
            python_edges,
            candidate_edges,
            matlab_endpoint_pairs,
        )
        if frontier_overlap_summary is not None:
            comparison["diagnostics"]["extra_frontier_missing_vertex_overlap"] = (
                frontier_overlap_summary
            )
    if candidate_audit is not None:
        comparison["diagnostics"]["candidate_audit"] = _candidate_audit_summary(candidate_audit)

    return comparison



def compare_networks(
    matlab_network: dict[str, Any],
    python_network: dict[str, Any],
    matlab_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare network-level statistics."""
    if (
        matlab_stats is None
        and "strands_to_vertices" not in matlab_network
        and "strands" not in matlab_network
    ):
        matlab_stats = matlab_network
        matlab_network = {}

    comparison = {
        "matlab_strand_count": _resolve_count(
            (matlab_stats or {}).get("strand_count"),
            _resolve_count(
                (matlab_network or {}).get("strand_count"),
                _count_items(
                    (matlab_network or {}).get(
                        "strands_to_vertices",
                        (matlab_network or {}).get("strands"),
                    )
                ),
            ),
        ),
        "python_strand_count": _infer_strand_count(python_network),
        "exact_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
    }

    matlab_count = comparison["matlab_strand_count"]
    python_count = comparison["python_strand_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["strand_count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["strand_count_percent_difference"] = (
                comparison["strand_count_difference"] / avg_count
            ) * 100.0

    matlab_counter = Counter(_strand_signatures(matlab_network or {}))
    python_counter = Counter(_strand_signatures(python_network))
    comparison["exact_match"] = matlab_counter == python_counter
    comparison["matlab_only_samples"] = _sample_counter_diff(matlab_counter, python_counter)
    comparison["python_only_samples"] = _sample_counter_diff(python_counter, matlab_counter)

    return comparison


def compare_results(
    matlab_results: dict[str, Any],
    python_results: dict[str, Any],
    matlab_parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare MATLAB and Python vectorization results.

    Args:
        matlab_results: Results from MATLAB execution (includes timing, paths, etc.)
        python_results: Results from Python execution (includes timing and processed data)
        matlab_parsed: Optional parsed MATLAB data from .mat files

    Returns:
        Comprehensive comparison dictionary
    """
    python_data = python_results.get("results") or {}
    matlab_vertices_count = _resolve_count(
        matlab_results.get("vertices_count"),
        _infer_vertices_count((matlab_parsed or {}).get("vertices", {})),
    )
    matlab_edges_count = _resolve_count(
        matlab_results.get("edges_count"),
        _infer_edges_count((matlab_parsed or {}).get("edges", {})),
    )
    matlab_strands_count = _resolve_count(
        matlab_results.get("strand_count"),
        _resolve_count(
            matlab_results.get("network_strands_count"),
            _resolve_count(
                (matlab_parsed or {}).get("network_stats", {}).get("strand_count"),
                _infer_strand_count((matlab_parsed or {}).get("network", {})),
            ),
        ),
    )
    python_vertices_count = _resolve_count(
        python_results.get("vertices_count"),
        _infer_vertices_count(python_data.get("vertices", {})),
    )
    python_edges_count = _resolve_count(
        python_results.get("edges_count"),
        _infer_edges_count(python_data.get("edges", {})),
    )
    python_strands_count = _resolve_count(
        python_results.get("network_strands_count"),
        _infer_strand_count(python_data.get("network", {})),
    )

    comparison = {
        "matlab": {
            "success": matlab_results.get("success", False),
            "elapsed_time": matlab_results.get("elapsed_time", 0.0),
            "output_dir": matlab_results.get("output_dir", ""),
            "vertices_count": matlab_vertices_count,
            "edges_count": matlab_edges_count,
            "strand_count": matlab_strands_count,
        },
        "python": {
            "success": python_results.get("success", False),
            "elapsed_time": python_results.get("elapsed_time", 0.0),
            "output_dir": python_results.get("output_dir", ""),
            "vertices_count": python_vertices_count,
            "edges_count": python_edges_count,
            "network_strands_count": python_strands_count,
            "comparison_mode": python_results.get("comparison_mode", {}),
        },
        "performance": {},
    }

    # Performance comparison
    matlab_time = matlab_results.get("elapsed_time", 0.0)
    python_time = python_results.get("elapsed_time", 0.0)

    if matlab_time > 0 and python_time > 0:
        speedup = matlab_time / python_time
        comparison["performance"] = {
            "matlab_time_seconds": matlab_time,
            "python_time_seconds": python_time,
            "speedup": speedup,
            "faster": "Python" if speedup > 1.0 else "MATLAB",
        }

    # Detailed comparison if parsed MATLAB data is available
    if matlab_parsed and python_data:
        # Compare vertices
        if "vertices" in matlab_parsed and "vertices" in python_data:
            comparison["vertices"] = compare_vertices(
                matlab_parsed["vertices"], python_data["vertices"]
            )

        # Compare edges
        if "edges" in matlab_parsed and "edges" in python_data:
            comparison["edges"] = compare_edges(
                matlab_parsed["edges"],
                python_data["edges"],
                python_results.get("candidate_edges") or python_data.get("candidate_edges"),
                python_results.get("candidate_audit") or python_data.get("candidate_audit"),
            )

        # Compare networks
        if "network" in matlab_parsed and "network" in python_data:
            comparison["network"] = compare_networks(
                matlab_parsed["network"],
                python_data["network"],
                matlab_parsed.get("network_stats"),
            )

    parity_gate = {
        "vertices_exact": comparison.get("vertices", {}).get("exact_positions_scales_match"),
        "edges_exact": comparison.get("edges", {}).get("exact_match"),
        "strands_exact": comparison.get("network", {}).get("exact_match"),
    }
    available_checks = [value for value in parity_gate.values() if value is not None]
    parity_gate["passed"] = all(available_checks) if available_checks else None
    comparison["parity_gate"] = parity_gate

    return comparison
