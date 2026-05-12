from __future__ import annotations

import numpy as np

from slavv_python.core.edges.selection import _choose_edges_matlab_style


def test_choose_edges_keeps_the_shared_neighborhood_frontier_partner():
    vertex_positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    vertex_scales = np.zeros(3, dtype=np.int16)
    frontier_partner_trace = np.array(
        [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 4.0, 1.0]],
        dtype=np.float32,
    )
    watershed_alternate_trace = np.array(
        [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0], [1.0, 4.0, 1.0], [1.0, 5.0, 1.0]],
        dtype=np.float32,
    )
    candidates = {
        "traces": [frontier_partner_trace, watershed_alternate_trace],
        "connections": np.array([[0, 1], [0, 2]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0], dtype=np.float32),
        ],
        "scale_traces": [
            np.zeros(3, dtype=np.int16),
            np.zeros(5, dtype=np.int16),
        ],
        "origin_indices": np.array([0, 0], dtype=np.int32),
        "connection_sources": ["frontier", "watershed"],
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert np.array_equal(chosen["traces"][0], frontier_partner_trace)
    assert chosen["connection_sources"] == ["frontier"]
    assert chosen["chosen_candidate_indices"].tolist() == [0]
    assert chosen["diagnostics"]["conflict_rejected_count"] == 1
    assert chosen["diagnostics"]["conflict_rejected_by_source"] == {"watershed": 1}
    assert chosen["diagnostics"]["conflict_blocking_source_counts"] == {"frontier": 1}
    assert chosen["diagnostics"]["conflict_source_pairs"] == {"watershed->frontier": 1}
