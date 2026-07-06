from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as ndi

from slavv_python.pipeline.edges.naive_watershed import (
    collect_naive_watershed_label_unit,
    paint_vertex_watershed_markers,
    run_skimage_watershed_labels,
)


@pytest.mark.unit
def test_collect_naive_watershed_label_unit_matches_shared_adjacency_logic():
    energy = np.ones((5, 5, 5), dtype=np.float32)
    vertex_positions = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    markers = paint_vertex_watershed_markers(vertex_positions, energy.shape)
    labels = run_skimage_watershed_labels(energy, markers, energy_sign=-1.0)
    structure = ndi.generate_binary_structure(3, 1)

    seen_pairs: set[tuple[int, int]] = set()
    unit_a = collect_naive_watershed_label_unit(
        1, labels, energy, structure, seen_pairs, coord_dtype=np.float32
    )
    unit_b = collect_naive_watershed_label_unit(
        2, labels, energy, structure, seen_pairs, coord_dtype=np.float64
    )

    assert unit_a.connections == [[0, 1]]
    assert unit_b.connections == []
    assert len(unit_a.traces) == 1
    assert unit_a.traces[0].dtype == np.float32
    assert np.isclose(unit_a.metrics[0], 1.0)
