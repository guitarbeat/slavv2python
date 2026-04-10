import numpy as np
from sklearn.neural_network import MLPClassifier

from slavv.analysis import MLCurator


def test_train_classifiers():
    curator = MLCurator()
    rng = np.random.default_rng(0)

    # Synthetic vertex data.
    Xv = rng.random((20, 5))
    yv = np.array([0, 1] * 10)
    res_v = curator.train_vertex_classifier(Xv, yv, method="matlab_nn")
    assert isinstance(curator.vertex_classifier, MLPClassifier)
    assert curator.vertex_classifier.activation == "logistic"
    assert "test_accuracy" in res_v

    # Synthetic edge data.
    Xe = rng.random((20, 4))
    ye = np.array([0, 1] * 10)
    res_e = curator.train_edge_classifier(Xe, ye, method="matlab_nn")
    assert isinstance(curator.edge_classifier, MLPClassifier)
    assert curator.edge_classifier.activation == "logistic"
    assert "test_accuracy" in res_e
