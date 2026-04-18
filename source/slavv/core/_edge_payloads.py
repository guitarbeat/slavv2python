"""Shared edge payload and diagnostics helpers for core SLAVV modules."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def _empty_stop_reason_counts() -> dict[str, int]:
    """Return the canonical edge-trace stop-reason counter payload."""
    return {
        "bounds": 0,
        "nan": 0,
        "energy_threshold": 0,
        "energy_rise_step_halving": 0,
        "max_steps": 0,
        "direct_terminal_hit": 0,
        "frontier_exhausted_nonnegative": 0,
        "length_limit": 0,
        "terminal_frontier_hit": 0,
    }


def build_edge_diagnostics(*, extra_fields: dict[str, object] | None = None) -> dict[str, Any]:
    """Return the canonical edge-diagnostics payload with optional extra fields."""
    diagnostics = {
        "candidate_traced_edge_count": 0,
        "terminal_edge_count": 0,
        "self_edge_count": 0,
        "duplicate_directed_pair_count": 0,
        "antiparallel_pair_count": 0,
        "chosen_edge_count": 0,
        "dangling_edge_count": 0,
        "negative_energy_rejected_count": 0,
        "conflict_rejected_count": 0,
        "conflict_rejected_by_source": {},
        "conflict_blocking_source_counts": {},
        "conflict_source_pairs": {},
        "degree_pruned_count": 0,
        "orphan_pruned_count": 0,
        "cycle_pruned_count": 0,
        "watershed_join_supplement_count": 0,
        "watershed_endpoint_degree_rejected": 0,
        "geodesic_join_supplement_count": 0,
        "terminal_direct_hit_count": 0,
        "terminal_reverse_center_hit_count": 0,
        "terminal_reverse_near_hit_count": 0,
        "stop_reason_counts": _empty_stop_reason_counts(),
    }
    if extra_fields:
        for key, value in extra_fields.items():
            if isinstance(value, dict):
                diagnostics[key] = dict(value)
            elif isinstance(value, list):
                diagnostics[key] = list(value)
            else:
                diagnostics[key] = value
    return cast("dict[str, Any]", diagnostics)


def _empty_edge_diagnostics() -> dict[str, Any]:
    """Return the canonical shared edge-diagnostics payload."""
    return build_edge_diagnostics()


def _merge_edge_diagnostics(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge additive edge diagnostics from one payload into another."""
    for key, value in source.items():
        if key == "stop_reason_counts":
            target_counts = target.setdefault("stop_reason_counts", _empty_stop_reason_counts())
            for stop_reason, count in value.items():
                target_counts[stop_reason] = int(target_counts.get(stop_reason, 0)) + int(count)
            continue

        if isinstance(value, dict):
            target_map = target.setdefault(key, {})
            if not isinstance(target_map, dict):
                target_map = {}
            for item_key, item_value in value.items():
                target_map[str(item_key)] = int(target_map.get(str(item_key), 0)) + int(item_value)
            target[key] = target_map
            continue

        if isinstance(value, (int, np.integer)):
            target[key] = int(target.get(key, 0)) + int(value)


def _empty_edges_result(vertex_positions: np.ndarray | None = None) -> dict[str, Any]:
    """Return the canonical empty edge payload."""
    positions = (
        np.asarray(vertex_positions, dtype=np.float32)
        if vertex_positions is not None
        else np.empty((0, 3), dtype=np.float32)
    )
    return cast(
        "dict[str, Any]",
        {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "energies": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "connection_sources": [],
            "vertex_positions": positions,
            "diagnostics": _empty_edge_diagnostics(),
            "chosen_candidate_indices": np.zeros((0,), dtype=np.int32),
        },
    )
