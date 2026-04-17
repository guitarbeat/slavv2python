"""Reproduction test for "Stingy Reachability" bottleneck."""

from __future__ import annotations

import numpy as np

from slavv.core.edge_candidates import _supplement_matlab_frontier_candidates_with_watershed_joins


def _empty_candidates() -> dict[str, object]:
    return {
        "connections": np.zeros((0, 2), dtype=np.int32),
        "traces": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
        },
    }


def _vertex_positions() -> np.ndarray:
    # Two vertices at (2,2,2) and (2,6,2)
    return np.array([[2.0, 2.0, 2.0], [2.0, 6.0, 2.0]], dtype=np.float32)


def test_reproduce_stingy_reachability_bottleneck() -> None:
    # Create energy such that there is a clear watershed path between vertices
    # but they have NO frontier candidates.
    energy = np.zeros((10, 10, 10), dtype=np.float32)
    energy[2, 2, 2] = -10.0
    energy[2, 3, 2] = -8.0
    energy[2, 4, 2] = -8.0
    energy[2, 5, 2] = -8.0
    energy[2, 6, 2] = -10.0
    # Rest of volume is higher energy
    energy[energy == 0] = -1.0

    # CASE 1: enforce_frontier_reachability=True (current default)
    # Neither vertex 0 nor vertex 1 is in frontier_vertices because _empty_candidates has no connections.
    result_enforced = _supplement_matlab_frontier_candidates_with_watershed_joins(
        _empty_candidates(),
        energy,
        None,
        _vertex_positions(),
        energy_sign=-1.0,  # Negative energy is good
        enforce_frontier_reachability=True,
    )

    # It should REJECT the join.
    assert result_enforced["diagnostics"]["watershed_reachability_rejected"] > 0
    assert result_enforced["diagnostics"]["watershed_accepted"] == 0

    # CASE 2: enforce_frontier_reachability=False (desired fix)
    result_relaxed = _supplement_matlab_frontier_candidates_with_watershed_joins(
        _empty_candidates(),
        energy,
        None,
        _vertex_positions(),
        energy_sign=-1.0,
        enforce_frontier_reachability=False,
    )

    # It should ACCEPT the join.
    assert result_relaxed["diagnostics"]["watershed_accepted"] == 1
    assert result_relaxed["diagnostics"]["watershed_reachability_rejected"] == 0
    assert len(result_relaxed["connections"]) == 1
