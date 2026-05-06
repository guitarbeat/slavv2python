import json

import numpy as np
from dev.tests.support.payload_builders import build_processing_results

from source.analysis.ml_curator import DrewsCurator, MLCurator


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


def test_generate_training_data_normalizes_results_and_uses_energy_image_shape(monkeypatch):
    curator = MLCurator()
    processing_results = [
        build_processing_results(
            overrides={"metadata": {"source": "typed-adapter"}},
            energy_data={
                "energy": np.zeros((6, 5, 4), dtype=np.float32),
                "scale_indices": np.zeros((6, 5, 4), dtype=np.int16),
                "image_shape": (6, 5, 4),
                "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
                "lumen_radius_microns": np.array([2.0], dtype=np.float32),
            },
        )
    ]
    manual_annotations = [
        {
            "vertex_labels": np.array([1, 0, 1], dtype=int),
            "edge_labels": np.array([1, 0], dtype=int),
        }
    ]

    seen_image_shapes: list[tuple[int, ...]] = []
    seen_vertex_payloads: list[dict[str, object]] = []
    seen_edge_payloads: list[dict[str, object]] = []

    def fake_extract_vertex_features(vertices, energy_data, image_shape):
        seen_vertex_payloads.append(vertices)
        seen_image_shapes.append(tuple(image_shape))
        assert "radii_microns" in vertices
        assert energy_data["image_shape"] == (6, 5, 4)
        return np.array([[1.0], [2.0], [3.0]], dtype=float)

    def fake_extract_edge_features(edges, vertices, energy_data):
        seen_edge_payloads.append(edges)
        assert "connections" in edges
        assert "positions" in vertices
        assert energy_data["image_shape"] == (6, 5, 4)
        return np.array([[4.0], [5.0]], dtype=float)

    monkeypatch.setattr(curator, "extract_vertex_features", fake_extract_vertex_features)
    monkeypatch.setattr(curator, "extract_edge_features", fake_extract_edge_features)

    vertex_features, vertex_labels, edge_features, edge_labels = curator.generate_training_data(
        processing_results,
        manual_annotations,
    )

    assert seen_image_shapes == [(6, 5, 4)]
    assert len(seen_vertex_payloads) == 1
    assert len(seen_edge_payloads) == 1
    assert vertex_features.shape == (3, 1)
    assert edge_features.shape == (2, 1)
    assert vertex_labels.tolist() == [1, 0, 1]
    assert edge_labels.tolist() == [1, 0]
