"""Regression tests for watershed-based parity candidate rejection rules."""

from __future__ import annotations

import numpy as np
import pytest

from slavv.core.edge_candidates import _supplement_matlab_frontier_candidates_with_watershed_joins

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def _empty_candidates() -> dict[str, object]:
    return {
        "connections": np.zeros((0, 2), dtype=np.int32),
        "traces": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {},
    }


def _vertex_positions() -> np.ndarray:
    return np.array([[2.0, 2.0, 2.0], [2.0, 6.0, 2.0]], dtype=np.float32)


def _run_watershed_supplement(
    energy: np.ndarray,
    *,
    energy_sign: float,
    enforce_frontier_reachability: bool,
) -> dict[str, object]:
    return _supplement_matlab_frontier_candidates_with_watershed_joins(
        _empty_candidates(),
        energy,
        None,
        _vertex_positions(),
        energy_sign=energy_sign,
        enforce_frontier_reachability=enforce_frontier_reachability,
    )


def test_watershed_supplement_rejects_unreachable_contacts() -> None:
    energy = np.zeros((10, 10, 10), dtype=np.float32)
    energy[2, 2, 2] = -10.0
    energy[2, 3, 2] = -8.0
    energy[2, 4, 2] = -8.0
    energy[2, 5, 2] = -8.0
    energy[2, 6, 2] = -10.0
    energy[energy == 0] = -1.0

    result = _run_watershed_supplement(
        energy,
        energy_sign=-1.0,
        enforce_frontier_reachability=True,
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["watershed_accepted"] == 0
    assert diagnostics["watershed_reachability_rejected"] > 0


def test_watershed_supplement_rejects_positive_sign_background_barrier() -> None:
    energy = np.ones((10, 10, 10), dtype=np.float32)
    energy[:, 4, :] = -0.5
    energy[2, 4, 2] = -0.5

    result = _run_watershed_supplement(
        energy,
        energy_sign=1.0,
        enforce_frontier_reachability=False,
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["watershed_energy_rejected"] > 0
    assert diagnostics["watershed_accepted"] == 0
