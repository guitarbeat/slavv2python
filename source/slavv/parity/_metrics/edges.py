from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .counts import _count_items, _infer_edges_count
from .signatures import (
    _candidate_endpoint_pair_details,
    _candidate_endpoint_pairs_by_source,
    _edge_endpoint_pair_set,
    _edge_endpoint_signatures,
    _edge_signatures,
    _incident_endpoint_pairs_by_vertex,
    _normalize_candidate_connection_sources,
    _sample_counter_diff,
    _sample_set_diff,
)


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
    candidate_endpoint_pair_set: set[tuple[int, int]] = set()
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
        subset = extra_frontier_entries[:threshold]
        if subset:
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

    matlab_total_length = matlab_edges.get("total_length", 0.0)
    if matlab_total_length > 0:
        comparison["total_length"]["matlab"] = float(matlab_total_length)

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
                matlab_endpoint_pairs, missing_pairs_by_vertex, candidate_edges
            ),
            "extra_candidate_seed_origin_samples": _extra_candidate_seed_origin_samples(
                matlab_endpoint_pairs, candidate_edges
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
            python_edges, candidate_edges, matlab_endpoint_pairs
        )
        if chosen_source_summary is not None:
            comparison["diagnostics"]["chosen_candidate_sources"] = chosen_source_summary
        frontier_overlap_summary = _frontier_missing_vertex_overlap_summary(
            python_edges, candidate_edges, matlab_endpoint_pairs
        )
        if frontier_overlap_summary is not None:
            comparison["diagnostics"]["extra_frontier_missing_vertex_overlap"] = (
                frontier_overlap_summary
            )
    if candidate_audit is not None:
        comparison["diagnostics"]["candidate_audit"] = _candidate_audit_summary(candidate_audit)

    return comparison
