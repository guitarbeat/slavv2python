"""Shared-neighborhood audit helpers for parity metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .signatures import (
    _candidate_endpoint_pair_details,
    _candidate_endpoint_pairs_by_source,
    _edge_endpoint_pair_set,
    _incident_endpoint_pairs_by_vertex,
)

_TRACKED_SHARED_NEIGHBORHOODS = (359, 866, 1283, 64)


def _final_endpoint_pairs_by_seed_origin(
    python_edges: dict[str, Any],
    candidate_edges: dict[str, Any],
) -> dict[int, set[tuple[int, int]]]:
    """Group final chosen endpoint pairs by the seed origin of the chosen candidate."""
    connections = np.asarray(
        candidate_edges.get("connections", np.zeros((0, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 2)
    if connections.size == 0:
        return {}

    origin_indices = np.asarray(
        candidate_edges.get("origin_indices", np.zeros((0,), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if origin_indices.size != len(connections):
        return {}

    chosen_candidate_indices = np.asarray(
        python_edges.get("chosen_candidate_indices", np.array([], dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    grouped: dict[int, set[tuple[int, int]]] = {}
    for candidate_index in chosen_candidate_indices:
        if candidate_index < 0 or candidate_index >= len(connections):
            continue
        origin_index = int(origin_indices[candidate_index])
        if origin_index < 0:
            continue
        start_vertex, end_vertex = (int(value) for value in connections[candidate_index][:2])
        if start_vertex < 0 or end_vertex < 0:
            continue
        endpoint_pair = (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )
        grouped.setdefault(origin_index, set()).add(endpoint_pair)
    return grouped


def _candidate_lifecycle_events_by_origin(
    candidate_lifecycle: dict[str, Any] | None,
) -> dict[int, list[dict[str, Any]]]:
    """Group lifecycle events by seed origin with deterministic ordering."""
    if not isinstance(candidate_lifecycle, dict):
        return {}

    grouped: dict[int, list[dict[str, Any]]] = {}
    for raw_event in candidate_lifecycle.get("events", []):
        if not isinstance(raw_event, dict):
            continue
        try:
            origin_index = int(raw_event.get("seed_origin_index", -1))
        except (TypeError, ValueError):
            continue
        if origin_index < 0:
            continue
        grouped.setdefault(origin_index, []).append(dict(raw_event))

    for origin_index, events in grouped.items():
        events.sort(
            key=lambda event: (
                int(event.get("terminal_hit_sequence", 0)),
                int(event.get("terminal_vertex_index", -1)),
                str(event.get("resolution_reason", "")),
            )
        )
        grouped[origin_index] = events
    return grouped


def _selection_sources_by_origin(
    coverage: dict[str, Any],
    present_origins: set[int],
    vertex_count: int,
) -> dict[int, list[str]]:
    """Return the reasons each shared-neighborhood origin was selected for audit."""
    selection_sources: dict[int, list[str]] = {}
    for origin_index in _TRACKED_SHARED_NEIGHBORHOODS:
        if origin_index in present_origins or (vertex_count > 0 and origin_index < vertex_count):
            selection_sources.setdefault(int(origin_index), []).append("tracked_hotspot")

    for entry in coverage.get("missing_matlab_seed_origin_samples", []):
        if not isinstance(entry, dict):
            continue
        try:
            origin_index = int(entry.get("seed_origin_index", -1))
        except (TypeError, ValueError):
            continue
        if origin_index >= 0:
            selection_sources.setdefault(origin_index, []).append("top_missing_seed_origin")

    for entry in coverage.get("extra_candidate_seed_origin_samples", []):
        if not isinstance(entry, dict):
            continue
        try:
            origin_index = int(entry.get("seed_origin_index", -1))
        except (TypeError, ValueError):
            continue
        if origin_index >= 0:
            selection_sources.setdefault(origin_index, []).append("top_extra_seed_origin")
    return selection_sources


def _first_shared_neighborhood_divergence(
    *,
    missing_candidate_pairs: list[tuple[int, int]],
    extra_candidate_pairs: list[tuple[int, int]],
    missing_final_pairs: list[tuple[int, int]],
    lifecycle_events: list[dict[str, Any]],
) -> tuple[str, str]:
    """Choose the earliest concrete divergence point for one neighborhood."""
    if missing_candidate_pairs:
        rejected_event = next(
            (
                event
                for event in lifecycle_events
                if not bool(event.get("survived_candidate_manifest"))
            ),
            None,
        )
        if rejected_event is not None:
            return (
                "pre_manifest_rejection",
                f"{rejected_event.get('resolution_reason', 'unknown')} at terminal "
                f"{int(rejected_event.get('terminal_vertex_index', -1))}",
            )
        if extra_candidate_pairs:
            return (
                "manifest_partner_substitution",
                f"missing {list(missing_candidate_pairs[0])} while extra {list(extra_candidate_pairs[0])} is emitted",
            )
        return (
            "candidate_admission_gap",
            f"missing candidate pair {list(missing_candidate_pairs[0])}",
        )

    if missing_final_pairs:
        emitted_not_chosen = next(
            (
                event
                for event in lifecycle_events
                if bool(event.get("survived_candidate_manifest"))
                and not bool(event.get("chosen_final_edge"))
            ),
            None,
        )
        if emitted_not_chosen is not None:
            return (
                "final_cleanup_loss",
                f"emitted {emitted_not_chosen.get('emitted_endpoint_pair')} but it was not retained",
            )
        return ("final_cleanup_loss", f"missing final pair {list(missing_final_pairs[0])}")

    if extra_candidate_pairs:
        return (
            "extra_candidate_survivor",
            f"extra candidate pair {list(extra_candidate_pairs[0])} has no MATLAB match",
        )

    return ("no_divergence_detected", "no neighborhood-level divergence detected")


def build_shared_neighborhood_audit(
    matlab_edges: dict[str, Any],
    python_edges: dict[str, Any],
    candidate_edges: dict[str, Any] | None,
    edge_comparison: dict[str, Any],
    candidate_audit: dict[str, Any] | None = None,
    candidate_lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Summarize the first neighborhood-level divergence for tracked and top origins."""
    if candidate_edges is None:
        return None

    coverage = edge_comparison.get("diagnostics", {}).get("candidate_endpoint_coverage", {})
    matlab_endpoint_pairs = _edge_endpoint_pair_set(matlab_edges)
    if not matlab_endpoint_pairs and not coverage:
        return None

    matlab_pairs_by_vertex = _incident_endpoint_pairs_by_vertex(matlab_endpoint_pairs)
    candidate_pairs_by_seed_origin, _, _, _ = _candidate_endpoint_pair_details(candidate_edges)
    final_pairs_by_seed_origin = _final_endpoint_pairs_by_seed_origin(python_edges, candidate_edges)
    lifecycle_events_by_origin = _candidate_lifecycle_events_by_origin(candidate_lifecycle)
    vertex_count = (
        int(candidate_audit.get("vertex_count", 0)) if isinstance(candidate_audit, dict) else 0
    )
    present_origins = (
        set(candidate_pairs_by_seed_origin)
        | set(final_pairs_by_seed_origin)
        | set(lifecycle_events_by_origin)
        | {
            int(entry.get("origin_index", -1))
            for entry in (candidate_audit or {}).get("per_origin_summary", [])
            if isinstance(entry, dict)
        }
    )
    selection_sources = _selection_sources_by_origin(coverage, present_origins, vertex_count)
    if not selection_sources:
        return None

    missing_seed_entries = {
        int(entry.get("seed_origin_index", -1)): entry
        for entry in coverage.get("missing_matlab_seed_origin_samples", [])
        if isinstance(entry, dict)
    }
    extra_seed_entries = {
        int(entry.get("seed_origin_index", -1)): entry
        for entry in coverage.get("extra_candidate_seed_origin_samples", [])
        if isinstance(entry, dict)
    }

    neighborhoods: list[dict[str, Any]] = []
    for origin_index in sorted(
        selection_sources,
        key=lambda origin: (
            -int(
                missing_seed_entries.get(origin, {}).get(
                    "missing_matlab_incident_endpoint_pair_count",
                    0,
                )
            ),
            int(origin),
        ),
    ):
        matlab_incident_pairs = sorted(matlab_pairs_by_vertex.get(origin_index, set()))
        candidate_pairs = sorted(candidate_pairs_by_seed_origin.get(origin_index, set()))
        final_pairs = sorted(final_pairs_by_seed_origin.get(origin_index, set()))
        lifecycle_events = lifecycle_events_by_origin.get(origin_index, [])
        missing_candidate_pairs = sorted(set(matlab_incident_pairs) - set(candidate_pairs))
        extra_candidate_pairs = sorted(set(candidate_pairs) - set(matlab_incident_pairs))
        missing_final_pairs = sorted(set(matlab_incident_pairs) - set(final_pairs))
        first_divergence_stage, first_divergence_reason = _first_shared_neighborhood_divergence(
            missing_candidate_pairs=missing_candidate_pairs,
            extra_candidate_pairs=extra_candidate_pairs,
            missing_final_pairs=missing_final_pairs,
            lifecycle_events=lifecycle_events,
        )
        neighborhoods.append(
            {
                "origin_index": int(origin_index),
                "selection_sources": selection_sources.get(origin_index, []),
                "matlab_incident_endpoint_pair_count": len(matlab_incident_pairs),
                "candidate_endpoint_pair_count": len(candidate_pairs),
                "final_chosen_endpoint_pair_count": len(final_pairs),
                "missing_matlab_incident_endpoint_pair_count": len(missing_candidate_pairs),
                "extra_candidate_endpoint_pair_count": len(extra_candidate_pairs),
                "missing_final_endpoint_pair_count": len(missing_final_pairs),
                "missing_matlab_incident_endpoint_pair_samples": [
                    list(pair) for pair in missing_candidate_pairs[:3]
                ],
                "candidate_endpoint_pair_samples": [list(pair) for pair in candidate_pairs[:3]],
                "final_chosen_endpoint_pair_samples": [list(pair) for pair in final_pairs[:3]],
                "extra_candidate_endpoint_pair_samples": [
                    list(pair) for pair in extra_candidate_pairs[:3]
                ],
                "first_divergence_stage": first_divergence_stage,
                "first_divergence_reason": first_divergence_reason,
                "lifecycle_summary": {
                    "terminal_hit_count": len(lifecycle_events),
                    "rejected_terminal_count": len(
                        [
                            event
                            for event in lifecycle_events
                            if not bool(event.get("survived_candidate_manifest"))
                        ]
                    ),
                    "emitted_candidate_count": len(
                        [
                            event
                            for event in lifecycle_events
                            if bool(event.get("survived_candidate_manifest"))
                        ]
                    ),
                    "chosen_final_edge_count": len(
                        [
                            event
                            for event in lifecycle_events
                            if bool(event.get("chosen_final_edge"))
                        ]
                    ),
                    "claim_reassignment_count": len(
                        [event for event in lifecycle_events if bool(event.get("claim_reassigned"))]
                    ),
                    "final_cleanup_loss_count": len(
                        [
                            event
                            for event in lifecycle_events
                            if bool(event.get("survived_candidate_manifest"))
                            and not bool(event.get("chosen_final_edge"))
                        ]
                    ),
                    "resolution_counts": dict(
                        Counter(
                            str(event.get("resolution_reason", "unknown"))
                            for event in lifecycle_events
                        )
                    ),
                },
                "lifecycle_event_samples": lifecycle_events[:5],
                "coverage_snapshot": {
                    "missing_seed_origin": missing_seed_entries.get(origin_index),
                    "extra_seed_origin": extra_seed_entries.get(origin_index),
                },
            }
        )

    top_neighborhood = neighborhoods[0] if neighborhoods else None
    return {
        "schema_version": 1,
        "tracked_hotspots": list(_TRACKED_SHARED_NEIGHBORHOODS),
        "selection_sources": {
            str(origin): reasons for origin, reasons in selection_sources.items()
        },
        "analyzed_origin_indices": [int(item["origin_index"]) for item in neighborhoods],
        "top_neighborhood": top_neighborhood,
        "neighborhoods": neighborhoods,
    }
