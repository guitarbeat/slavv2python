"""Shared supplement payload helpers for edge-candidate augmentation flows."""

from __future__ import annotations

from typing import Any

from slavv_python.pipeline.edges.candidate_manifest import (
    CandidateManifest,
    candidate_as_payload,
)
from slavv_python.pipeline.edges.payloads import _merge_edge_diagnostics


def _new_supplement_payload(
    candidate_source: str,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> CandidateManifest:
    """Build a normalized candidate supplement manifest shell."""
    return CandidateManifest.from_payload(
        {
            "candidate_source": candidate_source,
            "traces": [],
            "connections": [],
            "metrics": [],
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": [],
            "connection_sources": [],
            "diagnostics": {} if diagnostics is None else diagnostics,
        }
    )


def _append_supplement_row(
    manifest: CandidateManifest,
    *,
    pair: tuple[int, int],
    trace: Any,
    energy_trace: Any,
    scale_trace: Any,
    metric: float,
    origin_index: int,
    connection_source: str | None = None,
) -> None:
    """Append one accepted candidate row to a supplement manifest."""
    source_label = connection_source or str(manifest.extra.get("candidate_source", "unknown"))
    manifest.append_unit(
        {
            "candidate_source": source_label,
            "traces": [trace],
            "connections": [[pair[0], pair[1]]],
            "metrics": [metric],
            "energy_traces": [energy_trace],
            "scale_traces": [scale_trace],
            "origin_indices": [origin_index],
            "connection_sources": [source_label],
            "diagnostics": {},
        }
    )


def _increment_origin_count(
    diagnostics: dict[str, Any],
    origin_counts: dict[int, int],
    origin_index: int,
    *,
    key: str,
) -> int:
    """Increment and persist per-origin supplement counts."""
    origin_counts[origin_index] = origin_counts.get(origin_index, 0) + 1
    diagnostics.setdefault(key, {})[str(origin_index)] = int(origin_counts[origin_index])
    return origin_counts[origin_index]


def _merge_or_append_supplement(
    target: CandidateManifest | dict[str, Any],
    payload: CandidateManifest | dict[str, Any],
) -> None:
    """Append a supplement manifest or merge only its diagnostics when empty."""
    target_manifest = (
        target if isinstance(target, CandidateManifest) else CandidateManifest.from_payload(target)
    )
    payload_manifest = (
        payload
        if isinstance(payload, CandidateManifest)
        else CandidateManifest.from_payload(payload)
    )
    if payload_manifest.connections.size:
        target_manifest.append_unit(
            {
                **payload_manifest.extra,
                "traces": payload_manifest.traces,
                "connections": payload_manifest.connections,
                "metrics": payload_manifest.metrics,
                "energy_traces": payload_manifest.energy_traces,
                "scale_traces": payload_manifest.scale_traces,
                "origin_indices": payload_manifest.origin_indices,
                "connection_sources": payload_manifest.connection_sources,
                "frontier_lifecycle_events": payload_manifest.frontier_lifecycle_events,
                "diagnostics": payload_manifest.diagnostics,
            }
        )
    else:
        _merge_edge_diagnostics(target_manifest.diagnostics, payload_manifest.diagnostics)

    if isinstance(target, dict):
        target.clear()
        target.update(candidate_as_payload(target_manifest))
