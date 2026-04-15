import json

import numpy as np

from slavv.analysis.ml_curator import DrewsCurator, MLCurator


def test_aggregate_training_data_reads_npz_and_json(tmp_path):
    curator = MLCurator()

    np.savez(
        tmp_path / "chunk_a.npz",
        vertex_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        vertex_labels=np.array([1, 0], dtype=int),
        edge_features=np.array([[0.2, 0.3]], dtype=float),
        edge_labels=np.array([1], dtype=int),
    )

    payload = {
        "vertex_features": [[5.0, 6.0]],
        "vertex_labels": [1],
        "edge_features": [[0.4, 0.5]],
        "edge_labels": [0],
    }
    (tmp_path / "chunk_b.json").write_text(json.dumps(payload), encoding="utf-8")

    v_feat, v_lab, e_feat, e_lab = curator.aggregate_training_data(tmp_path, file_pattern="chunk_*")

    assert v_feat.shape == (3, 2)
    assert e_feat.shape == (2, 2)
    assert v_lab.tolist() == [1, 0, 1]
    assert e_lab.tolist() == [1, 0]


def test_drews_curator_filters_short_and_tortuous_edges():
    vertices = {
        "positions": np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=float),
        "radii_pixels": np.array([1.0, 1.0, 1.0], dtype=float),
    }
    edges = {
        "traces": [
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float),  # keep
            np.array([[0, 0, 0], [0.5, 0, 0]], dtype=float),  # too short
            np.array([[0, 0, 0], [0, 3, 0], [3, 0, 0]], dtype=float),  # high tortuosity
        ],
        "connections": [(0, 1), (0, 1), (0, 1)],
        "energies": np.array([0.9, 0.8, 0.7], dtype=float),
    }

    curator = DrewsCurator(min_length_radius_ratio=2.0, max_tortuosity=1.2, max_endpoint_gap=5.0)
    curated = curator.curate(edges, vertices)

    assert len(curated["traces"]) == 1
    assert len(curated["connections"]) == 1
    assert curated["original_indices"].tolist() == [0]
    assert curated["energies"].tolist() == [0.9]
