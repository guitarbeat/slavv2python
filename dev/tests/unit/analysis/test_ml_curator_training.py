import json

import numpy as np
from slavv.analysis.ml_curator_training import load_aggregated_training_data


def test_load_aggregated_training_data_reads_npz_and_json(tmp_path):
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

    v_feat, v_lab, e_feat, e_lab = load_aggregated_training_data(tmp_path, file_pattern="chunk_*")

    assert v_feat.shape == (3, 2)
    assert e_feat.shape == (2, 2)
    assert v_lab.tolist() == [1, 0, 1]
    assert e_lab.tolist() == [1, 0]


def test_load_aggregated_training_data_returns_four_empty_arrays_for_empty_dir(tmp_path):
    v_feat, v_lab, e_feat, e_lab = load_aggregated_training_data(tmp_path / "missing")

    assert v_feat.shape == (0,)
    assert v_lab.shape == (0,)
    assert e_feat.shape == (0,)
    assert e_lab.shape == (0,)
