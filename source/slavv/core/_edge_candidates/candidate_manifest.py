"""Candidate manifest append helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .._edge_payloads import _merge_edge_diagnostics
from .audit import _normalize_candidate_connection_sources


def _append_candidate_unit(target: dict[str, Any], unit_payload: dict[str, Any]) -> None:
    """Append a per-origin candidate payload into the aggregate candidate manifest."""
    unit_traces = [np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"]]
    unit_connections = np.asarray(unit_payload["connections"], dtype=np.int32).reshape(-1, 2)
    unit_metrics = np.asarray(unit_payload["metrics"], dtype=np.float32).reshape(-1)
    unit_origin_indices = np.asarray(
        unit_payload.get("origin_indices", []), dtype=np.int32
    ).reshape(-1)
    unit_connection_sources = _normalize_candidate_connection_sources(
        unit_payload.get("connection_sources"),
        len(unit_connections),
        default_source=str(unit_payload.get("candidate_source", "unknown")),
    )
    target["traces"].extend(unit_traces)
    target["energy_traces"].extend(
        np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
    )
    target["scale_traces"].extend(
        np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
    )

    if unit_connections.size:
        target["connections"] = (
            unit_connections
            if target["connections"].size == 0
            else np.vstack([target["connections"], unit_connections])
        )
        target["metrics"] = np.concatenate([target["metrics"], unit_metrics])
        target["origin_indices"] = np.concatenate([target["origin_indices"], unit_origin_indices])
        target.setdefault("connection_sources", []).extend(unit_connection_sources)

    _merge_edge_diagnostics(
        cast("dict[str, Any]", target["diagnostics"]),
        cast("dict[str, Any]", unit_payload.get("diagnostics", {})),
    )
