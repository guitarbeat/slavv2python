"""Unit tests for strict synthetic dual-run compare / ladder stop predicate."""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.analytics.parity.probes.synthetic_dual_run_compare import (
    NonComparableArtifactsError,
    count_matlab_strands2vertices,
    first_break_surface,
    first_diff_stage,
    localize_stage_compare,
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


# --- U1: MATLAB strands2vertices counting (AE1) ---


@pytest.mark.unit
def test_ae1_numeric_3x2_matlab_strands_count_as_three():
    """Live double_junction_32 mats are numeric (3, 2); must not count as 1."""
    strands = np.asarray([[1, 2], [2, 3], [1, 3]], dtype=np.uint8)
    assert count_matlab_strands2vertices(strands) == 3


@pytest.mark.unit
def test_matlab_object_cell_strands_count_as_three():
    cell = np.empty((3,), dtype=object)
    cell[0] = np.asarray([1, 2], dtype=np.uint8)
    cell[1] = np.asarray([2, 3], dtype=np.uint8)
    cell[2] = np.asarray([1, 3], dtype=np.uint8)
    assert count_matlab_strands2vertices(cell) == 3


@pytest.mark.unit
def test_matlab_single_row_or_vector_strands_count_as_one():
    assert count_matlab_strands2vertices(np.asarray([[4, 7]], dtype=np.int64)) == 1
    assert count_matlab_strands2vertices(np.asarray([4, 7], dtype=np.int64)) == 1


@pytest.mark.unit
def test_matlab_missing_strands_count_is_none():
    assert count_matlab_strands2vertices(None) is None


# --- U2: stage localization + endpoint multisets ---


@pytest.mark.unit
def test_strand_endpoint_multiset_mismatch_is_first_diff_strands():
    matlab = _side([[2, 2, 2], [5, 5, 5], [8, 8, 8]], [[1, 2], [2, 3]], 2)
    python = _side([[1, 1, 1], [4, 4, 4], [7, 7, 7]], [[0, 1], [1, 2]], 2)
    # Counts match (2) but endpoint pairs differ after 0-based normalize.
    matlab["strands2vertices"] = np.asarray([[1, 2], [2, 3]], dtype=np.int64)
    python["strands"] = [np.asarray([0, 2]), np.asarray([1, 2])]  # ends (0,2),(1,2)
    assert first_diff_stage(matlab, python) == "strands"


@pytest.mark.unit
def test_first_diff_all_match_returns_none():
    matlab = _side([[2, 2, 2], [5, 5, 5]], [[1, 2]], 1)
    python = _side([[1, 1, 1], [4, 4, 4]], [[0, 1]], 1)
    matlab["strands2vertices"] = np.asarray([[1, 2]], dtype=np.int64)
    python["strands"] = [np.asarray([0, 1])]
    assert first_diff_stage(matlab, python) is None
    loc = localize_stage_compare(matlab, python, previously_reported_strand_break=True)
    assert loc["outcome"] == "measurement_fixed_match"
    assert "NOT Certification" in loc["note"]
    assert loc["stages"]["candidates"] == "unavailable"


@pytest.mark.unit
def test_candidates_unavailable_still_compares_edges():
    matlab = _side(
        [[2, 2, 2], [5, 5, 5], [8, 8, 8]],
        [[1, 2], [2, 3]],
        2,
    )
    python = _side(
        [[1, 1, 1], [4, 4, 4], [7, 7, 7]],
        [[0, 1]],  # missing edge
        2,
    )
    # Only Python has candidates — must not invent a candidates verdict.
    python["candidate_connections"] = np.asarray([[0, 1]], dtype=np.int64)
    assert first_diff_stage(matlab, python) == "edges"


@pytest.mark.unit
def test_ae3_candidates_not_compared_to_matlab_finals():
    """Python candidates vs MATLAB finals must not become a discovery residual."""
    matlab = _side([[2, 2, 2], [5, 5, 5]], [[1, 2]], 1)
    python = _side([[1, 1, 1], [4, 4, 4]], [[0, 1]], 1)
    matlab["strands2vertices"] = np.asarray([[1, 2]], dtype=np.int64)
    python["strands"] = [np.asarray([0, 1])]
    python["candidate_connections"] = np.asarray([[0, 1], [0, 0]], dtype=np.int64)
    loc = localize_stage_compare(matlab, python)
    assert loc["stages"]["candidates"] == "unavailable"
    assert loc["first_diff_stage"] is None
    assert "never compared to MATLAB final" in loc["same_class_guard"]
