"""Unit tests for strict synthetic dual-run compare / ladder stop predicate."""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.analytics.parity.probes.synthetic_dual_run_compare import (
    NonComparableArtifactsError,
    first_break_surface,
    strict_compare_summary,
)


def _side(
    positions: list[list[float]],
    connections: list[list[int]],
    n_strands: int,
    *,
    ok: bool = True,
) -> dict:
    return {
        "ok": ok,
        "positions": np.asarray(positions, dtype=np.float64),
        "connections": np.asarray(connections, dtype=np.int64),
        "n_strands": n_strands,
    }


@pytest.mark.unit
def test_full_match_returns_none():
    # MATLAB 1-based positions/connections; Python 0-based positions, 0-based conns.
    matlab = _side([[2, 2, 2], [5, 5, 5]], [[1, 2]], 1)
    python = _side([[1, 1, 1], [4, 4, 4]], [[0, 1]], 1)
    assert first_break_surface(matlab, python) is None
    summary = strict_compare_summary(matlab, python)
    assert summary["comparable"] is True
    assert summary["match"] is True
    assert summary["first_break_surface"] is None


@pytest.mark.unit
def test_ae1_edges_first_break_when_vertices_match():
    matlab = _side(
        [[2, 2, 2], [5, 5, 5], [8, 8, 8]],
        [[1, 2], [2, 3]],
        2,
    )
    python = _side(
        [[1, 1, 1], [4, 4, 4], [7, 7, 7]],
        [[0, 1]],  # missing second edge pair
        2,
    )
    assert first_break_surface(matlab, python) == "edges"


@pytest.mark.unit
def test_vertices_break_before_edges_even_if_pairs_also_differ():
    matlab = _side([[2, 2, 2], [5, 5, 5]], [[1, 2]], 1)
    python = _side([[1, 1, 1], [9, 9, 9]], [[0, 1]], 1)
    assert first_break_surface(matlab, python) == "vertices"


@pytest.mark.unit
def test_strands_break_when_vertices_and_pairs_match():
    matlab = _side([[2, 2, 2], [5, 5, 5]], [[1, 2]], 2)
    python = _side([[1, 1, 1], [4, 4, 4]], [[0, 1]], 1)
    assert first_break_surface(matlab, python) == "strands"


@pytest.mark.unit
def test_empty_both_sides_is_match():
    matlab = _side([], [], 0)
    python = _side([], [], 0)
    assert first_break_surface(matlab, python) is None


@pytest.mark.unit
def test_zero_edges_with_matching_vertices_is_match():
    matlab = _side([[2, 2, 2]], [], 0)
    python = _side([[1, 1, 1]], [], 0)
    matlab["connections"] = np.zeros((0, 2), dtype=np.int64)
    python["connections"] = np.zeros((0, 2), dtype=np.int64)
    assert first_break_surface(matlab, python) is None


@pytest.mark.unit
def test_incomplete_artifacts_raise_and_summary_non_comparable():
    matlab = {"ok": True, "positions": np.zeros((1, 3))}
    python = _side([[0, 0, 0]], [[0, 0]], 0)
    with pytest.raises(NonComparableArtifactsError):
        first_break_surface(matlab, python)
    summary = strict_compare_summary(matlab, python)
    assert summary["comparable"] is False
    assert summary["first_break_surface"] is None
