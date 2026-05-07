"""Candidate audit and normalization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from .common import Int32Array

from .common import normalize_candidate_connection_sources as _normalize_connection_sources


def _normalize_candidate_origin_counts(raw_counts: dict[Any, Any] | None) -> dict[str, int]:
    """Return a JSON-safe mapping from origin index to candidate count."""
    normalized: dict[str, int] = {}
    if not raw_counts:
        return normalized

    for key, value in raw_counts.items():
        try:
            normalized[str(int(key))] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_candidate_count_map(raw_counts: dict[Any, Any] | None) -> dict[str, int]:
    """Return a JSON-safe additive counter map with string keys."""
    normalized: dict[str, int] = {}
    if not raw_counts:
        return normalized

    for key, value in raw_counts.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        try:
            normalized[key_text] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
    *,
    default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection slavv_python label list."""
    return _normalize_connection_sources(
        raw_sources,
        candidate_connection_count,
        default_source=default_source,
    )


def _collect_candidate_source_maps(
    connections: Int32Array,
    origin_indices: Int32Array,
    connection_sources: list[str],
) -> tuple[
    dict[int, int],
    dict[int, set[tuple[int, int]]],
    dict[str, set[tuple[int, int]]],
    dict[tuple[int, int], set[str]],
    dict[str, dict[int, set[tuple[int, int]]]],
    dict[str, set[int]],
]:
    """Collect per-origin and per-source endpoint-pair maps for audit."""
    total_origin_counts: dict[int, int] = {}
    total_origin_pairs: dict[int, set[tuple[int, int]]] = {}
    source_pair_sets: dict[str, set[tuple[int, int]]] = {
        "frontier": set(),
        "watershed": set(),
        "geodesic": set(),
        "fallback": set(),
    }
    pair_sources: dict[tuple[int, int], set[str]] = {}
    source_origin_pair_sets: dict[str, dict[int, set[tuple[int, int]]]] = {
        "frontier": {},
        "watershed": {},
        "geodesic": {},
        "fallback": {},
    }
    source_origin_sets: dict[str, set[int]] = {
        "frontier": set(),
        "watershed": set(),
        "geodesic": set(),
        "fallback": set(),
    }

    for index, origin_index in enumerate(origin_indices):
        origin_index_int = int(origin_index)
        if origin_index_int < 0:
            continue
        total_origin_counts[origin_index_int] = total_origin_counts.get(origin_index_int, 0) + 1

        source_label = connection_sources[index] if index < len(connection_sources) else "unknown"
        if source_label in source_origin_sets:
            source_origin_sets[source_label].add(origin_index_int)

        if index >= len(connections):
            continue
        start_vertex, end_vertex = (int(value) for value in connections[index][:2])
        if start_vertex < 0 or end_vertex < 0:
            continue
        endpoint_pair = (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )
        total_origin_pairs.setdefault(origin_index_int, set()).add(endpoint_pair)
        if source_label in source_pair_sets:
            pair_sources.setdefault(endpoint_pair, set()).add(source_label)
            source_pair_sets[source_label].add(endpoint_pair)
            source_origin_pair_sets[source_label].setdefault(origin_index_int, set()).add(
                endpoint_pair
            )
    return (
        total_origin_counts,
        total_origin_pairs,
        source_pair_sets,
        pair_sources,
        source_origin_pair_sets,
        source_origin_sets,
    )


def _build_origin_payload_rows(
    *,
    total_origin_counts: dict[int, int],
    total_origin_pairs: dict[int, set[tuple[int, int]]],
    source_origin_pair_sets: dict[str, dict[int, set[tuple[int, int]]]],
    frontier_origin_counts: dict[int, int],
    supplement_origin_counts: dict[int, int],
    geodesic_origin_counts_int: dict[int, int],
    frontier_terminal_hits_int: dict[int, int],
    frontier_terminal_accepts_int: dict[int, int],
    frontier_terminal_rejections_int: dict[int, int],
) -> list[dict[str, Any]]:
    """Build the per-origin audit payload rows."""
    per_origin_payload: list[dict[str, Any]] = []
    all_origins = (
        set(total_origin_counts.keys())
        | set(frontier_origin_counts.keys())
        | set(supplement_origin_counts.keys())
        | set(geodesic_origin_counts_int.keys())
        | set(frontier_terminal_hits_int.keys())
        | set(frontier_terminal_accepts_int.keys())
        | set(frontier_terminal_rejections_int.keys())
    )
    for origin_index in sorted(all_origins):
        frontier_count = int(frontier_origin_counts.get(origin_index, 0))
        supplement_count = int(supplement_origin_counts.get(origin_index, 0))
        geodesic_count = int(geodesic_origin_counts_int.get(origin_index, 0))
        frontier_terminal_hit_count = int(frontier_terminal_hits_int.get(origin_index, 0))
        frontier_terminal_accept_count = int(frontier_terminal_accepts_int.get(origin_index, 0))
        frontier_terminal_rejection_count = int(
            frontier_terminal_rejections_int.get(origin_index, 0)
        )
        total_count = int(total_origin_counts.get(origin_index, 0))
        fallback_count = max(0, total_count - frontier_count - supplement_count - geodesic_count)
        candidate_pairs = total_origin_pairs.get(origin_index, set())
        frontier_pairs = source_origin_pair_sets["frontier"].get(origin_index, set())
        watershed_pairs = source_origin_pair_sets["watershed"].get(origin_index, set())
        geodesic_pairs = source_origin_pair_sets["geodesic"].get(origin_index, set())
        fallback_pairs = source_origin_pair_sets["fallback"].get(origin_index, set())
        per_origin_payload.append(
            {
                "origin_index": origin_index,
                "frontier_candidate_count": frontier_count,
                "watershed_candidate_count": supplement_count,
                "geodesic_candidate_count": geodesic_count,
                "fallback_candidate_count": fallback_count,
                "frontier_terminal_hit_count": frontier_terminal_hit_count,
                "frontier_terminal_accept_count": frontier_terminal_accept_count,
                "frontier_terminal_rejection_count": frontier_terminal_rejection_count,
                "candidate_connection_count": total_count,
                "candidate_endpoint_pair_count": len(candidate_pairs),
                "candidate_endpoint_pair_samples": sorted(candidate_pairs)[:3],
                "frontier_endpoint_pair_count": len(frontier_pairs),
                "frontier_endpoint_pair_samples": sorted(frontier_pairs)[:3],
                "watershed_endpoint_pair_count": len(watershed_pairs),
                "watershed_endpoint_pair_samples": sorted(watershed_pairs)[:3],
                "geodesic_endpoint_pair_count": len(geodesic_pairs),
                "geodesic_endpoint_pair_samples": sorted(geodesic_pairs)[:3],
                "fallback_endpoint_pair_count": len(fallback_pairs),
                "fallback_endpoint_pair_samples": sorted(fallback_pairs)[:3],
            }
        )
    return per_origin_payload


def _build_candidate_diagnostics(diag: Any) -> dict[str, int]:
    """Return the flattened audit diagnostics counters."""
    return {
        "candidate_traced_edge_count": int(diag.get("candidate_traced_edge_count", 0)),
        "terminal_edge_count": int(diag.get("terminal_edge_count", 0)),
        "chosen_edge_count": int(diag.get("chosen_edge_count", 0)),
        "watershed_join_supplement_count": int(diag.get("watershed_join_supplement_count", 0)),
        "watershed_endpoint_degree_rejected": int(
            diag.get("watershed_endpoint_degree_rejected", 0)
        ),
        "watershed_total_pairs": int(diag.get("watershed_total_pairs", 0)),
        "watershed_already_existing": int(diag.get("watershed_already_existing", 0)),
        "watershed_short_trace_rejected": int(diag.get("watershed_short_trace_rejected", 0)),
        "watershed_energy_rejected": int(diag.get("watershed_energy_rejected", 0)),
        "watershed_metric_threshold_rejected": int(
            diag.get("watershed_metric_threshold_rejected", 0)
        ),
        "watershed_reachability_rejected": int(diag.get("watershed_reachability_rejected", 0)),
        "watershed_mutual_frontier_rejected": int(
            diag.get("watershed_mutual_frontier_rejected", 0)
        ),
        "watershed_cap_rejected": int(diag.get("watershed_cap_rejected", 0)),
        "watershed_accepted": int(diag.get("watershed_accepted", 0)),
        "geodesic_join_supplement_count": int(diag.get("geodesic_join_supplement_count", 0)),
        "geodesic_route_failed": int(diag.get("geodesic_route_failed", 0)),
        "geodesic_energy_rejected": int(diag.get("geodesic_energy_rejected", 0)),
        "geodesic_metric_threshold_rejected": int(
            diag.get("geodesic_metric_threshold_rejected", 0)
        ),
        "geodesic_path_ratio_rejected": int(diag.get("geodesic_path_ratio_rejected", 0)),
        "geodesic_vertex_crossing_rejected": int(diag.get("geodesic_vertex_crossing_rejected", 0)),
        "geodesic_endpoint_degree_rejected": int(diag.get("geodesic_endpoint_degree_rejected", 0)),
        "geodesic_origin_budget_rejected": int(diag.get("geodesic_origin_budget_rejected", 0)),
        "geodesic_shared_neighborhood_endpoint_relaxed": int(
            diag.get("geodesic_shared_neighborhood_endpoint_relaxed", 0)
        ),
        "geodesic_accepted": int(diag.get("geodesic_accepted", 0)),
        "frontier_origins_with_candidates": int(diag.get("frontier_origins_with_candidates", 0)),
        "frontier_origins_without_candidates": int(
            diag.get("frontier_origins_without_candidates", 0)
        ),
    }


def _build_pair_source_breakdown(
    pair_sources: dict[tuple[int, int], set[str]],
) -> dict[str, Any]:
    """Build the pair-source overlap summary."""
    frontier_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"frontier"}
    )
    watershed_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"watershed"}
    )
    fallback_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"fallback"}
    )
    geodesic_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"geodesic"}
    )
    multi_source_pairs = sorted(pair for pair, sources in pair_sources.items() if len(sources) > 1)
    return {
        "frontier_only_pair_count": len(frontier_only_pairs),
        "watershed_only_pair_count": len(watershed_only_pairs),
        "geodesic_only_pair_count": len(geodesic_only_pairs),
        "fallback_only_pair_count": len(fallback_only_pairs),
        "multi_source_pair_count": len(multi_source_pairs),
        "frontier_only_endpoint_pair_samples": frontier_only_pairs[:5],
        "watershed_only_endpoint_pair_samples": watershed_only_pairs[:5],
        "geodesic_only_endpoint_pair_samples": geodesic_only_pairs[:5],
        "fallback_only_endpoint_pair_samples": fallback_only_pairs[:5],
    }


def _build_edge_candidate_audit(
    candidates: dict[str, Any],
    vertex_count: int,
    use_frontier_tracer: bool,
    frontier_origin_counts: dict[int, int] | None = None,
    supplement_origin_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Build a stable, JSON-serializable summary of edge-candidate provenance."""
    connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32
    )
    if connections.ndim == 1 and connections.size > 0:
        connections = connections.reshape(-1, 2)
    candidate_connection_count = int(connections.shape[0]) if connections.size else 0

    origin_indices = np.asarray(
        candidates.get("origin_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32
    ).reshape(-1)
    if origin_indices.size != candidate_connection_count:
        origin_indices = np.zeros((candidate_connection_count,), dtype=np.int32)

    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        candidate_connection_count,
        default_source=str(candidates.get("candidate_source", "unknown")),
    )

    (
        total_origin_counts,
        total_origin_pairs,
        source_pair_sets,
        pair_sources,
        source_origin_pair_sets,
        source_origin_sets,
    ) = _collect_candidate_source_maps(
        cast("Int32Array", connections),
        cast("Int32Array", origin_indices),
        connection_sources,
    )

    frontier_origin_counts = {
        int(origin): int(count) for origin, count in (frontier_origin_counts or {}).items()
    }
    supplement_origin_counts = {
        int(origin): int(count) for origin, count in (supplement_origin_counts or {}).items()
    }
    diag = candidates.get("diagnostics", {})
    geodesic_origin_counts = _normalize_candidate_origin_counts(
        diag.get("geodesic_per_origin_candidate_counts") if isinstance(diag, dict) else None
    )
    geodesic_origin_counts_int = {
        int(origin): int(count) for origin, count in geodesic_origin_counts.items()
    }
    frontier_terminal_hits = _normalize_candidate_origin_counts(
        diag.get("frontier_per_origin_terminal_hits") if isinstance(diag, dict) else None
    )
    frontier_terminal_accepts = _normalize_candidate_origin_counts(
        diag.get("frontier_per_origin_terminal_accepts") if isinstance(diag, dict) else None
    )
    frontier_terminal_rejections = _normalize_candidate_origin_counts(
        diag.get("frontier_per_origin_terminal_rejections") if isinstance(diag, dict) else None
    )
    frontier_terminal_hits_int = {
        int(origin): int(count) for origin, count in frontier_terminal_hits.items()
    }
    frontier_terminal_accepts_int = {
        int(origin): int(count) for origin, count in frontier_terminal_accepts.items()
    }
    frontier_terminal_rejections_int = {
        int(origin): int(count) for origin, count in frontier_terminal_rejections.items()
    }
    frontier_connection_count = (
        sum(frontier_origin_counts.values())
        if frontier_origin_counts
        else len([src for src in connection_sources if src == "frontier"])
    )
    supplement_connection_count = (
        sum(supplement_origin_counts.values())
        if supplement_origin_counts
        else len([src for src in connection_sources if src == "watershed"])
    )
    geodesic_connection_count = len([src for src in connection_sources if src == "geodesic"])
    fallback_connection_count = max(
        0,
        candidate_connection_count
        - frontier_connection_count
        - supplement_connection_count
        - geodesic_connection_count,
    )
    frontier_origin_count = (
        len(frontier_origin_counts)
        if frontier_origin_counts
        else len(source_origin_sets["frontier"])
    )
    supplement_origin_count = (
        len(supplement_origin_counts)
        if supplement_origin_counts
        else len(source_origin_sets["watershed"])
    )
    geodesic_origin_count = (
        len(geodesic_origin_counts_int)
        if geodesic_origin_counts_int
        else len(source_origin_sets["geodesic"])
    )
    fallback_origin_count = max(
        0,
        len(total_origin_counts)
        - len(
            source_origin_sets["frontier"]
            | source_origin_sets["watershed"]
            | source_origin_sets["geodesic"]
        ),
    )

    per_origin_payload = _build_origin_payload_rows(
        total_origin_counts=total_origin_counts,
        total_origin_pairs=total_origin_pairs,
        source_origin_pair_sets=source_origin_pair_sets,
        frontier_origin_counts=frontier_origin_counts,
        supplement_origin_counts=supplement_origin_counts,
        geodesic_origin_counts_int=geodesic_origin_counts_int,
        frontier_terminal_hits_int=frontier_terminal_hits_int,
        frontier_terminal_accepts_int=frontier_terminal_accepts_int,
        frontier_terminal_rejections_int=frontier_terminal_rejections_int,
    )
    candidate_diagnostics = _build_candidate_diagnostics(diag)

    fallback_source_total = {
        "candidate_connection_count": fallback_connection_count,
        "candidate_origin_count": fallback_origin_count,
        "candidate_endpoint_pair_count": len(source_pair_sets["fallback"]),
        "candidate_endpoint_pair_samples": sorted(source_pair_sets["fallback"])[:5],
    }
    return {
        "schema_version": 1,
        "vertex_count": vertex_count,
        "use_frontier_tracer": use_frontier_tracer,
        "candidate_traces": len(candidates.get("traces", [])),
        "candidate_connection_count": candidate_connection_count,
        "candidate_origin_count": len(total_origin_counts),
        "source_breakdown": {
            "frontier": {
                "candidate_connection_count": frontier_connection_count,
                "candidate_origin_count": frontier_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["frontier"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["frontier"])[:5],
            },
            "watershed": {
                "candidate_connection_count": supplement_connection_count,
                "candidate_origin_count": supplement_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["watershed"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["watershed"])[:5],
            },
            "geodesic": {
                "candidate_connection_count": geodesic_connection_count,
                "candidate_origin_count": geodesic_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["geodesic"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["geodesic"])[:5],
            },
            "fallback": fallback_source_total,
        },
        "frontier_per_origin_candidate_counts": frontier_origin_counts,
        "frontier_per_origin_terminal_hits": frontier_terminal_hits,
        "frontier_per_origin_terminal_accepts": frontier_terminal_accepts,
        "frontier_per_origin_terminal_rejections": frontier_terminal_rejections,
        "watershed_per_origin_candidate_counts": _normalize_candidate_origin_counts(
            diag.get("watershed_per_origin_candidate_counts")
        ),
        "geodesic_per_origin_candidate_counts": geodesic_origin_counts,
        "frontier_terminal_resolution_counts": _normalize_candidate_count_map(
            diag.get("frontier_terminal_resolution_counts")
        ),
        "pair_source_breakdown": _build_pair_source_breakdown(pair_sources),
        "per_origin_summary": per_origin_payload,
        "diagnostic_counters": candidate_diagnostics,
    }
