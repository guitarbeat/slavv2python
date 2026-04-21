from __future__ import annotations

import numpy as np

from slavv.core._edge_candidates.watershed_contacts import (
    _augment_matlab_frontier_candidates_with_watershed_contacts,
)
from slavv.core.edge_candidates import (
    _finalize_matlab_parity_candidates,
    _matlab_frontier_edge_budget,
    _matlab_parity_frontier_budget_mode,
    _params_with_matlab_parity_edge_budget,
    _params_with_matlab_parity_frontier_budget_mode,
    _params_with_matlab_parity_watershed_candidate_mode,
    _parity_watershed_candidate_mode,
)
from slavv.core.edge_selection import choose_edges_for_workflow


def test_matlab_frontier_edge_budget_prefers_internal_parity_override():
    params = _params_with_matlab_parity_edge_budget(
        {"comparison_exact_network": True, "number_of_edges_per_vertex": 4},
        edge_budget=2,
    )

    assert _matlab_frontier_edge_budget(params) == 2
    assert _matlab_frontier_edge_budget({"number_of_edges_per_vertex": 4}) == 4


def test_finalize_matlab_parity_candidates_uses_forced_matlab_budget(monkeypatch):
    captured: dict[str, int | str] = {}

    def fake_augment(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        *,
        max_edges_per_vertex,
        candidate_mode,
        parity_watershed_metric_threshold,
    ):
        del energy, scale_indices, vertex_positions, energy_sign
        captured["candidate_mode"] = candidate_mode
        del parity_watershed_metric_threshold
        captured["max_edges_per_vertex"] = max_edges_per_vertex
        return candidates

    monkeypatch.setattr(
        "slavv.core._edge_candidates.generate._augment_matlab_frontier_candidates_with_watershed_contacts",
        fake_augment,
    )

    params = _params_with_matlab_parity_edge_budget(
        {
            "comparison_exact_network": True,
            "number_of_edges_per_vertex": 4,
            "parity_candidate_salvage_mode": "none",
        },
        edge_budget=2,
    )
    params = _params_with_matlab_parity_watershed_candidate_mode(
        params,
        candidate_mode="remaining_origin_contacts",
    )
    candidates = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": {},
    }

    _finalize_matlab_parity_candidates(
        candidates,
        np.zeros((3, 3, 3), dtype=np.float32),
        np.zeros((3, 3, 3), dtype=np.int16),
        np.zeros((0, 3), dtype=np.float32),
        -1.0,
        params,
        np.ones(3, dtype=np.float32),
    )

    assert captured["max_edges_per_vertex"] == 2
    assert captured["candidate_mode"] == "remaining_origin_contacts"


def test_parity_watershed_candidate_mode_prefers_internal_override():
    params = _params_with_matlab_parity_watershed_candidate_mode(
        {"comparison_exact_network": True, "parity_watershed_candidate_mode": "all_contacts"},
        candidate_mode="remaining_origin_contacts",
    )

    assert _parity_watershed_candidate_mode(params) == "remaining_origin_contacts"
    assert _parity_watershed_candidate_mode(
        {"parity_watershed_candidate_mode": "all_contacts"}
    ) == ("all_contacts")


def test_matlab_parity_frontier_budget_mode_prefers_internal_override():
    params = _params_with_matlab_parity_frontier_budget_mode(
        {
            "comparison_exact_network": True,
            "parity_frontier_budget_mode": "terminal_hits",
        },
        mode="accepted_candidates",
    )

    assert _matlab_parity_frontier_budget_mode(params) == "accepted_candidates"
    assert _matlab_parity_frontier_budget_mode({}) == "terminal_hits"


def test_remaining_origin_contacts_rejects_watershed_pair_when_frontier_origin_is_at_budget(
    monkeypatch,
):
    def fake_build_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        parity_watershed_metric_threshold=None,
    ):
        del candidates
        del energy
        del scale_indices
        del vertex_positions
        del energy_sign
        del parity_watershed_metric_threshold
        row = (
            (0, 3),
            np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 7.0]], dtype=np.float32),
            np.array([-3.0, -3.0], dtype=np.float32),
            np.zeros(2, dtype=np.int16),
            -3.0,
            2.0,
        )
        return [row], {
            "watershed_total_pairs": 1,
            "watershed_already_existing": 0,
            "watershed_short_trace_rejected": 0,
            "watershed_energy_rejected": 0,
            "watershed_metric_threshold_rejected": 0,
        }

    monkeypatch.setattr(
        "slavv.core._edge_candidates.watershed_contacts._build_watershed_candidate_rows",
        fake_build_rows,
    )

    candidates = {
        "traces": [
            np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 5.0], [5.0, 7.0, 5.0]], dtype=np.float32),
            np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [7.0, 5.0, 5.0]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [0, 2]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16), np.zeros(3, dtype=np.int16)],
        "origin_indices": np.array([0, 0], dtype=np.int32),
        "connection_sources": ["frontier", "frontier"],
        "diagnostics": {},
    }

    result = _augment_matlab_frontier_candidates_with_watershed_contacts(
        candidates,
        np.zeros((9, 9, 9), dtype=np.float32),
        None,
        np.zeros((4, 3), dtype=np.float32),
        -1.0,
        max_edges_per_vertex=2,
        candidate_mode="remaining_origin_contacts",
    )

    assert result["connections"].tolist() == [[0, 1], [0, 2]]
    assert result["diagnostics"]["watershed_join_supplement_count"] == 0
    assert result["diagnostics"]["watershed_origin_budget_rejected"] == 1


def test_origin_cap_allows_watershed_pairs_even_when_frontier_origin_is_at_budget(monkeypatch):
    def fake_build_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        parity_watershed_metric_threshold=None,
    ):
        del candidates
        del energy
        del scale_indices
        del vertex_positions
        del energy_sign
        del parity_watershed_metric_threshold
        rows = [
            (
                (0, 3),
                np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 7.0]], dtype=np.float32),
                np.array([-3.0, -3.0], dtype=np.float32),
                np.zeros(2, dtype=np.int16),
                -3.0,
                2.0,
            ),
            (
                (0, 4),
                np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 8.0]], dtype=np.float32),
                np.array([-2.5, -2.5], dtype=np.float32),
                np.zeros(2, dtype=np.int16),
                -2.5,
                3.0,
            ),
            (
                (0, 5),
                np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 9.0]], dtype=np.float32),
                np.array([-2.0, -2.0], dtype=np.float32),
                np.zeros(2, dtype=np.int16),
                -2.0,
                4.0,
            ),
        ]
        return rows, {
            "watershed_total_pairs": 3,
            "watershed_already_existing": 0,
            "watershed_short_trace_rejected": 0,
            "watershed_energy_rejected": 0,
            "watershed_metric_threshold_rejected": 0,
        }

    monkeypatch.setattr(
        "slavv.core._edge_candidates.watershed_contacts._build_watershed_candidate_rows",
        fake_build_rows,
    )

    candidates = {
        "traces": [
            np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 5.0], [5.0, 7.0, 5.0]], dtype=np.float32),
            np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [7.0, 5.0, 5.0]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [0, 2]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16), np.zeros(3, dtype=np.int16)],
        "origin_indices": np.array([0, 0], dtype=np.int32),
        "connection_sources": ["frontier", "frontier"],
        "diagnostics": {},
    }

    result = _augment_matlab_frontier_candidates_with_watershed_contacts(
        candidates,
        np.zeros((10, 10, 10), dtype=np.float32),
        None,
        np.zeros((6, 3), dtype=np.float32),
        -1.0,
        max_edges_per_vertex=2,
        candidate_mode="origin_cap",
    )

    assert result["connections"].tolist() == [[0, 1], [0, 2], [0, 3], [0, 4]]
    assert result["diagnostics"]["watershed_join_supplement_count"] == 2
    assert result["diagnostics"]["watershed_origin_budget_rejected"] == 1


def test_choose_edges_for_workflow_uses_forced_matlab_budget_in_exact_network_cleanup():
    vertex_positions = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 7.0, 5.0],
            [7.0, 5.0, 5.0],
            [5.0, 5.0, 7.0],
        ],
        dtype=np.float32,
    )
    traces = [
        np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 5.0], [5.0, 7.0, 5.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [7.0, 5.0, 5.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 6.0], [5.0, 5.0, 7.0]], dtype=np.float32),
    ]
    candidates = {
        "traces": traces,
        "connections": np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0, -3.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0], dtype=np.float32),
            np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16) for _ in range(3)],
        "origin_indices": np.array([0, 0, 0], dtype=np.int32),
        "connection_sources": ["frontier", "frontier", "frontier"],
        "diagnostics": {},
    }
    params = _params_with_matlab_parity_edge_budget(
        {"comparison_exact_network": True, "number_of_edges_per_vertex": 4},
        edge_budget=2,
    )

    chosen = choose_edges_for_workflow(
        candidates,
        vertex_positions,
        np.zeros(4, dtype=np.int16),
        np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        (12, 12, 12),
        params,
    )

    assert chosen["connections"].tolist() == [[0, 1], [0, 2]]
    assert chosen["chosen_candidate_indices"].tolist() == [0, 1]
    assert chosen["diagnostics"]["degree_pruned_count"] == 1
