import numpy as np

from slavv.core._edge_candidates.watershed_candidates import _build_watershed_candidate_rows
from slavv.core._edge_payloads import _empty_edge_diagnostics


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
