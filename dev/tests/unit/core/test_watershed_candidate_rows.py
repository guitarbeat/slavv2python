import numpy as np
from source.core._edge_candidates.generate import _generate_edge_candidates
from source.core._edge_candidates.watershed_candidates import (
    _augment_candidates_with_watershed_contacts,
    _build_watershed_candidate_rows,
    _parity_watershed_candidate_mode,
)
from source.core._edge_payloads import _empty_edge_diagnostics


def test_build_watershed_candidate_rows_tracks_common_rejections_and_preserves_pair_order():
    energy = np.ones((9, 9, 9), dtype=np.float64)
    energy[4, 1:4, 4] = -1.0
    energy[4, 5:8, 4] = -0.5
    vertex_positions = np.array(
        [
            [4.0, 1.0, 4.0],
            [4.0, 3.0, 4.0],
            [4.0, 5.0, 4.0],
            [4.0, 7.0, 4.0],
        ],
        dtype=np.float32,
    )
    candidates = {
        "connections": np.array([[0, 1]], dtype=np.int32),
        "diagnostics": _empty_edge_diagnostics(),
    }

    rows, diagnostics = _build_watershed_candidate_rows(
        candidates,
        energy,
        None,
        vertex_positions,
        -1.0,
    )

    assert diagnostics["watershed_already_existing"] >= 1
    assert diagnostics["watershed_total_pairs"] >= len(rows)
    assert diagnostics["watershed_energy_rejected"] >= 1
    assert diagnostics["watershed_short_trace_rejected"] >= 0

    assert rows
    pairs = [pair for pair, *_rest in rows]
    assert pairs == sorted(pairs)

    _, threshold_diagnostics = _build_watershed_candidate_rows(
        candidates,
        energy,
        None,
        vertex_positions,
        -1.0,
        metric_threshold=-0.75,
    )
    assert threshold_diagnostics["watershed_metric_threshold_rejected"] >= 1


def test_augment_candidates_with_watershed_contacts_appends_missing_pairs_and_tracks_sources(
    monkeypatch,
):
    def fake_build_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        metric_threshold=None,
    ):
        del candidates, energy, scale_indices, vertex_positions, energy_sign, metric_threshold
        return [
            (
                (0, 2),
                np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 1.0]], dtype=np.float32),
                np.array([-3.0, -3.0], dtype=np.float32),
                np.zeros(2, dtype=np.int16),
                -3.0,
                2.0,
            )
        ], {
            "watershed_total_pairs": 1,
            "watershed_already_existing": 0,
            "watershed_short_trace_rejected": 0,
            "watershed_energy_rejected": 0,
            "watershed_metric_threshold_rejected": 0,
        }

    monkeypatch.setattr(
        "source.core._edge_candidates.watershed_candidates._build_watershed_candidate_rows",
        fake_build_rows,
    )

    candidates = {
        "traces": [np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float32)],
        "connections": np.array([[0, 1]], dtype=np.int32),
        "metrics": np.array([-5.0], dtype=np.float32),
        "energy_traces": [np.array([-5.0, -5.0], dtype=np.float32)],
        "scale_traces": [np.zeros(2, dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "connection_sources": ["fallback"],
        "diagnostics": _empty_edge_diagnostics(),
    }

    augmented = _augment_candidates_with_watershed_contacts(
        candidates,
        np.zeros((5, 5, 5), dtype=np.float32),
        None,
        np.zeros((3, 3), dtype=np.float32),
        -1.0,
        candidate_mode="all_contacts",
        max_edges_per_vertex=4,
    )

    assert augmented["connections"].tolist() == [[0, 1], [0, 2]]
    assert augmented["connection_sources"] == ["fallback", "watershed"]
    assert augmented["diagnostics"]["watershed_join_supplement_count"] == 1
    assert augmented["diagnostics"]["watershed_accepted"] == 1
    assert augmented["diagnostics"]["watershed_per_origin_candidate_counts"] == {"0": 1}


def test_augment_candidates_with_watershed_contacts_respects_remaining_origin_budget(
    monkeypatch,
):
    def fake_build_rows(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        metric_threshold=None,
    ):
        del candidates, energy, scale_indices, vertex_positions, energy_sign, metric_threshold
        return [
            (
                (0, 3),
                np.array([[1.0, 1.0, 1.0], [1.0, 4.0, 1.0]], dtype=np.float32),
                np.array([-3.0, -3.0], dtype=np.float32),
                np.zeros(2, dtype=np.int16),
                -3.0,
                3.0,
            )
        ], {
            "watershed_total_pairs": 1,
            "watershed_already_existing": 0,
            "watershed_short_trace_rejected": 0,
            "watershed_energy_rejected": 0,
            "watershed_metric_threshold_rejected": 0,
        }

    monkeypatch.setattr(
        "source.core._edge_candidates.watershed_candidates._build_watershed_candidate_rows",
        fake_build_rows,
    )

    candidates = {
        "traces": [
            np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [0, 2]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(2, dtype=np.int16), np.zeros(2, dtype=np.int16)],
        "origin_indices": np.array([0, 0], dtype=np.int32),
        "connection_sources": ["fallback", "fallback"],
        "diagnostics": _empty_edge_diagnostics(),
    }

    augmented = _augment_candidates_with_watershed_contacts(
        candidates,
        np.zeros((5, 5, 5), dtype=np.float32),
        None,
        np.zeros((4, 3), dtype=np.float32),
        -1.0,
        candidate_mode="remaining_origin_contacts",
        max_edges_per_vertex=2,
    )

    assert augmented["connections"].tolist() == [[0, 1], [0, 2]]
    assert augmented["diagnostics"]["watershed_join_supplement_count"] == 0
    assert augmented["diagnostics"]["watershed_origin_budget_rejected"] == 1


def test_generate_edge_candidates_applies_watershed_supplement_for_exact_network(monkeypatch):
    captured: dict[str, float | int | str | None] = {}

    def fake_trace_origin(**kwargs):
        del kwargs
        return (
            [np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float32)],
            [[0, 1]],
            [-5.0],
            [np.array([-5.0, -5.0], dtype=np.float32)],
            [np.zeros(2, dtype=np.int16)],
            [0],
            ["fallback"],
        )

    def fake_augment(
        candidates,
        energy,
        scale_indices,
        vertex_positions,
        energy_sign,
        *,
        max_edges_per_vertex,
        candidate_mode,
        metric_threshold,
    ):
        del energy, scale_indices, vertex_positions, energy_sign, candidates
        captured["max_edges_per_vertex"] = max_edges_per_vertex
        captured["candidate_mode"] = candidate_mode
        captured["metric_threshold"] = metric_threshold
        return {"connections": np.array([[0, 1], [0, 2]], dtype=np.int32)}

    monkeypatch.setattr(
        "source.core._edge_candidates.generate._trace_fallback_origin_candidates",
        fake_trace_origin,
    )
    monkeypatch.setattr(
        "source.core._edge_candidates.generate._augment_candidates_with_watershed_contacts",
        fake_augment,
    )

    result = _generate_edge_candidates(
        np.zeros((5, 5, 5), dtype=np.float32),
        None,
        np.zeros((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.int16),
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.zeros((5, 5, 5), dtype=np.int16),
        None,
        0.0,
        {
            "comparison_exact_network": True,
            "parity_watershed_candidate_mode": "all_contacts",
            "parity_watershed_metric_threshold": -3.5,
            "number_of_edges_per_vertex": 4,
        },
        -1.0,
    )

    assert result["connections"].tolist() == [[0, 1], [0, 2]]
    assert captured == {
        "max_edges_per_vertex": 4,
        "candidate_mode": "all_contacts",
        "metric_threshold": -3.5,
    }


def test_parity_watershed_candidate_mode_maps_legacy_mode_to_all_contacts():
    assert (
        _parity_watershed_candidate_mode(
            {
                "comparison_exact_network": True,
                "parity_watershed_candidate_mode": "legacy_supplement",
            }
        )
        == "all_contacts"
    )



