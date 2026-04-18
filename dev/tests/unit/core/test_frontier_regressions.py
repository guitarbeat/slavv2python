from __future__ import annotations

import numpy as np

from slavv.core.edge_candidates import _trace_origin_edges_matlab_frontier
from slavv.core.vertices import paint_vertex_center_image


def test_rejected_child_better_than_parent_does_not_seed_invalid_parent_record(monkeypatch):
    """A rejected child should not become a synthetic invalid parent path.

    This keeps later shared-neighborhood branches from inheriting a `(-1, -1)`
    parent record on the next terminal hit.
    """

    energy = np.full((9, 9, 9), 1.0, dtype=np.float32)
    energy[4, 4, 4] = -9.0
    energy[4, 5, 4] = -8.0
    energy[4, 6, 4] = -7.0
    energy[5, 4, 4] = -6.0
    energy[6, 4, 4] = -5.0
    energy[4, 3, 4] = -4.0
    energy[4, 2, 4] = -3.0

    vertex_positions = np.array(
        [
            [4.0, 4.0, 4.0],
            [4.0, 6.0, 4.0],
            [6.0, 4.0, 4.0],
            [4.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )
    vertex_scales = np.zeros(4, dtype=np.int16)
    center_image = paint_vertex_center_image(vertex_positions, energy.shape)

    resolve_calls = {"count": 0}

    def fake_resolve(
        _path_linear,
        _terminal_vertex_idx,
        _seed_origin_idx,
        edge_paths_linear,
        edge_pairs,
        _pointer_index_map,
        _energy,
        _shape,
    ):
        resolve_calls["count"] += 1
        if resolve_calls["count"] == 1:
            return (
                None,
                None,
                "rejected_child_better_than_parent",
                {
                    "parent_path_max_energy": -4.0,
                    "child_path_max_energy": -6.0,
                },
            )

        assert edge_paths_linear == []
        assert edge_pairs == []
        return (
            0,
            2,
            "accepted_parent_origin_half",
            {"bifurcation_choice": "parent_origin_half"},
        )

    monkeypatch.setattr(
        "slavv.core.edge_candidates._resolve_frontier_edge_connection_details",
        fake_resolve,
    )

    payload = _trace_origin_edges_matlab_frontier(
        energy,
        np.zeros_like(energy, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        np.array([1.0], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        center_image,
        0,
        {
            "number_of_edges_per_vertex": 2,
            "space_strel_apothem": 1,
            "max_edge_length_per_origin_radius": 8.0,
        },
    )

    assert resolve_calls["count"] == 2
    assert payload["connections"] == [[0, 2]]
    assert payload["frontier_lifecycle_events"][0]["resolution_reason"] == (
        "rejected_child_better_than_parent"
    )
