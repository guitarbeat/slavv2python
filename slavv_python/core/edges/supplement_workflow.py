"""Shared supplement payload helpers for edge-candidate augmentation flows."""

from __future__ import annotations

from typing import Any

from slavv_python.core.edges.candidate_manifest import _append_candidate_unit
from slavv_python.core.edges.payloads import _merge_edge_diagnostics


def _new_supplement_payload(
    candidate_source: str,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized candidate supplement payload shell."""
    return {
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


def _append_supplement_row(
    payload: dict[str, Any],
    *,
    pair: tuple[int, int],
    trace: Any,
    energy_trace: Any,
    scale_trace: Any,
    metric: float,
    origin_index: int,
    connection_source: str | None = None,
) -> None:
    """Append one accepted candidate row to a supplement payload."""
    source_label = connection_source or str(payload.get("candidate_source", "unknown"))
    payload["traces"].append(trace)
    payload["connections"].append([pair[0], pair[1]])
    payload["metrics"].append(metric)
    payload["energy_traces"].append(energy_trace)
    payload["scale_traces"].append(scale_trace)
    payload["origin_indices"].append(origin_index)
    payload["connection_sources"].append(source_label)


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


def _merge_or_append_supplement(target: dict[str, Any], payload: dict[str, Any]) -> None:
    """Append a supplement payload or merge only its diagnostics when empty."""
    if payload["connections"]:
        _append_candidate_unit(target, payload)
        return
    _merge_edge_diagnostics(target.get("diagnostics", {}), payload.get("diagnostics", {}))
