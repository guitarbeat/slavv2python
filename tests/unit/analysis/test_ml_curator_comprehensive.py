"""Consolidated and comprehensive tests for ML Curator feature extraction, training, security, and IO."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tests.support.payload_builders import build_processing_results

from slavv_python.analysis import MLCurator
from slavv_python.analysis.ml_curator import DrewsCurator
from slavv_python.analysis.ml_curator_features import (
    compute_local_gradient,
    feature_importance,
    in_bounds,
)
from slavv_python.analysis.ml_curator_io import materialize_model_source
from slavv_python.analysis.ml_curator_training import load_aggregated_training_data

if TYPE_CHECKING:
    from Any = object


class _UploadedModel:
    def __init__(self, path: Path):
        self.name = path.name
        self._payload = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._payload


class _Malicious:
    def __reduce__(self) -> tuple:
        return (os.system, ("echo malicious code execution",))


# ==============================================================================
# Feature Extraction Tests
# ==============================================================================


@pytest.mark.unit
def test_compute_local_gradient_uses_central_difference():
    energy = np.zeros((5, 5, 5), dtype=float)
    energy[3, 2, 2] = 3.0
    energy[1, 2, 2] = 1.0
    energy[2, 3, 2] = 8.0
    energy[2, 1, 2] = 2.0
    energy[2, 2, 3] = 5.0
    energy[2, 2, 1] = 1.0

    gradient = compute_local_gradient(energy, np.array([2.0, 2.0, 2.0]))

    assert gradient.tolist() == [1.0, 3.0, 2.0]


@pytest.mark.unit
def test_in_bounds_checks_all_dimensions():
    assert in_bounds(np.array([1, 2, 3]), (2, 3, 4)) is True
    assert in_bounds(np.array([2, 2, 3]), (2, 3, 4)) is False


@pytest.mark.unit
def test_feature_importance_returns_none_when_missing():
    class _NoImportance:
        pass

    class _WithImportance:
        feature_importances_ = np.array([0.1, 0.9])

    assert feature_importance(_NoImportance()) is None
    assert feature_importance(_WithImportance()).tolist() == [0.1, 0.9]


# ==============================================================================
# DrewsCurator Tests
# ==============================================================================


@pytest.mark.unit
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


# ==============================================================================
# Model Security Tests
# ==============================================================================


@pytest.mark.unit
def test_ml_curator_load_malicious_model(tmp_path: Path):
    filepath = tmp_path / "malicious_model.pkl"
    with filepath.open("wb") as f:
        pickle.dump(_Malicious(), f)

    curator = MLCurator()

    with pytest.raises(pickle.UnpicklingError, match=r"forbidden|Failed to safely unpickle"):
        curator.load_models(vertex_path=filepath)

    with pytest.raises(pickle.UnpicklingError, match=r"forbidden|Failed to safely unpickle"):
        curator.load_models(edge_path=filepath)


# ==============================================================================
# Aggregation and Training Data Generation Tests
# ==============================================================================


@pytest.mark.unit
def test_load_aggregated_training_data_reads_npz_and_json(tmp_path: Path):
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


@pytest.mark.unit
def test_load_aggregated_training_data_returns_four_empty_arrays_for_empty_dir(tmp_path: Path):
    v_feat, v_lab, e_feat, e_lab = load_aggregated_training_data(tmp_path / "missing")

    assert v_feat.shape == (0,)
    assert v_lab.shape == (0,)
    assert e_feat.shape == (0,)
    assert e_lab.shape == (0,)


@pytest.mark.unit
def test_aggregate_training_data_reads_npz_and_json(tmp_path: Path):
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


@pytest.mark.unit
def test_generate_training_data_uses_annotation_edge_labels():
    curator = MLCurator()
    results = [
        {
            "vertices": {
                "positions": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
                "energies": np.array([0.5], dtype=np.float32),
                "scales": np.array([0], dtype=np.int16),
                "radii_pixels": np.array([1.0], dtype=np.float32),
                "radii_microns": np.array([1.0], dtype=np.float32),
            },
            "edges": {
                "traces": [np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)],
                "energies": np.array([0.25], dtype=np.float32),
                "connections": np.array([[0, 0]], dtype=np.int32),
            },
            "energy_data": {
                "energy": np.zeros((4, 4, 4), dtype=np.float32),
                "scale_indices": np.zeros((4, 4, 4), dtype=np.int16),
                "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
            },
            "image_shape": (4, 4, 4),
        }
    ]
    annotations = [{"vertex_labels": np.array([1]), "edge_labels": np.array([0])}]

    _, vertex_labels, _, edge_labels = curator.generate_training_data(results, annotations)

    np.testing.assert_array_equal(vertex_labels, np.array([1]))
    np.testing.assert_array_equal(edge_labels, np.array([0]))


@pytest.mark.unit
def test_generate_training_data_normalizes_results_and_uses_energy_image_shape(monkeypatch):
    curator = MLCurator()
    processing_results = [
        build_processing_results(
            overrides={"metadata": {"slavv_python": "typed-adapter"}},
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


# ==============================================================================
# Classifier Training Tests
# ==============================================================================


@pytest.mark.unit
def test_train_classifiers():
    curator = MLCurator()
    rng = np.random.default_rng(0)

    Xv = rng.random((20, 5))
    yv = np.array([0, 1] * 10)
    res_v = curator.train_vertex_classifier(Xv, yv, method="single_hidden_layer_mlp")
    assert isinstance(curator.vertex_classifier, MLPClassifier)
    assert curator.vertex_classifier.activation == "logistic"
    assert "test_accuracy" in res_v

    Xe = rng.random((20, 4))
    ye = np.array([0, 1] * 10)
    res_e = curator.train_edge_classifier(Xe, ye, method="single_hidden_layer_mlp")
    assert isinstance(curator.edge_classifier, MLPClassifier)
    assert curator.edge_classifier.activation == "logistic"
    assert "test_accuracy" in res_e


# ==============================================================================
# IO and Materialization Tests
# ==============================================================================


@pytest.mark.unit
def test_save_and_load_models(tmp_path: Path):
    curator = MLCurator()
    rng = np.random.default_rng(0)

    Xv = rng.random((20, 3))
    yv = rng.integers(0, 2, size=20)
    curator.vertex_scaler.fit(Xv)
    curator.vertex_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.vertex_classifier.fit(curator.vertex_scaler.transform(Xv), yv)

    Xe = rng.random((20, 3))
    ye = rng.integers(0, 2, size=20)
    curator.edge_scaler.fit(Xe)
    curator.edge_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.edge_classifier.fit(curator.edge_scaler.transform(Xe), ye)

    vertex_path = tmp_path / "vertex.joblib"
    edge_path = tmp_path / "edge.joblib"
    curator.save_models(vertex_path, edge_path)

    loaded = MLCurator()
    loaded.load_models(vertex_path, edge_path)

    X_test_v = rng.random((5, 3))
    orig_v = curator.vertex_classifier.predict(curator.vertex_scaler.transform(X_test_v))
    new_v = loaded.vertex_classifier.predict(loaded.vertex_scaler.transform(X_test_v))
    assert np.array_equal(orig_v, new_v)

    X_test_e = rng.random((5, 3))
    orig_e = curator.edge_classifier.predict(curator.edge_scaler.transform(X_test_e))
    new_e = loaded.edge_classifier.predict(loaded.edge_scaler.transform(X_test_e))
    assert np.array_equal(orig_e, new_e)


@pytest.mark.unit
def test_load_models_accepts_uploaded_file_objects(tmp_path: Path):
    curator = MLCurator()
    rng = np.random.default_rng(0)

    Xv = rng.random((20, 3))
    yv = rng.integers(0, 2, size=20)
    curator.vertex_scaler.fit(Xv)
    curator.vertex_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.vertex_classifier.fit(curator.vertex_scaler.transform(Xv), yv)

    Xe = rng.random((20, 3))
    ye = rng.integers(0, 2, size=20)
    curator.edge_scaler.fit(Xe)
    curator.edge_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.edge_classifier.fit(curator.edge_scaler.transform(Xe), ye)

    vertex_path = tmp_path / "vertex.joblib"
    edge_path = tmp_path / "edge.joblib"
    curator.save_models(vertex_path, edge_path)

    loaded = MLCurator()
    loaded.load_models(_UploadedModel(vertex_path), _UploadedModel(edge_path))

    assert loaded.vertex_classifier is not None
    assert loaded.edge_classifier is not None


@pytest.mark.unit
def test_materialize_model_source_supports_uploaded_file_objects(tmp_path: Path):
    source_path = tmp_path / "model.joblib"
    source_path.write_bytes(b"model-bytes")

    with materialize_model_source(_UploadedModel(source_path)) as materialized_path:
        assert Path(materialized_path).read_bytes() == b"model-bytes"
