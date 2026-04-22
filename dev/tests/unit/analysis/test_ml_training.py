import numpy as np
from sklearn.neural_network import MLPClassifier

from slavv.analysis import MLCurator


def test_train_classifiers():
    curator = MLCurator()
    rng = np.random.default_rng(0)

    # Synthetic vertex data.
    Xv = rng.random((20, 5))
    yv = np.array([0, 1] * 10)
    res_v = curator.train_vertex_classifier(Xv, yv, method="single_hidden_layer_mlp")
    assert isinstance(curator.vertex_classifier, MLPClassifier)
    assert curator.vertex_classifier.activation == "logistic"
    assert "test_accuracy" in res_v

    # Synthetic edge data.
    Xe = rng.random((20, 4))
    ye = np.array([0, 1] * 10)
    res_e = curator.train_edge_classifier(Xe, ye, method="single_hidden_layer_mlp")
    assert isinstance(curator.edge_classifier, MLPClassifier)
    assert curator.edge_classifier.activation == "logistic"
    assert "test_accuracy" in res_e


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
