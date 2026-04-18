"""Persisted edge unit loading helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from pathlib import Path


def _load_edge_units(
    units_dir: Path,
    append_candidate_unit: Any,
    empty_edge_diagnostics: Any,
) -> tuple[dict[str, Any], set[int]]:
    payload = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "frontier_lifecycle_events": [],
        "diagnostics": empty_edge_diagnostics(),
    }
    completed: set[int] = set()

    if not units_dir.exists():
        return payload, completed

    for unit_file in sorted(units_dir.glob("*.pkl")):
        unit_payload = safe_load(unit_file)
        origin_index = int(unit_payload["origin_index"])
        completed.add(origin_index)
        append_candidate_unit(payload, unit_payload)
    return payload, completed
