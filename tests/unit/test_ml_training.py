import sys
import pathlib
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from slavv.ml_curator import MLCurator


def test_train_classifiers():
    curator = MLCurator()
    # synthetic vertex data
    Xv = np.random.rand(20, 5)
    yv = np.array([0, 1] * 10)
    res_v = curator.train_vertex_classifier(Xv, yv, method='matlab_nn')
    assert isinstance(curator.vertex_classifier, MLPClassifier)
    assert curator.vertex_classifier.activation == 'logistic'
    assert 'test_accuracy' in res_v

    # synthetic edge data
    Xe = np.random.rand(20, 4)
    ye = np.array([0, 1] * 10)
    res_e = curator.train_edge_classifier(Xe, ye, method='matlab_nn')
    assert isinstance(curator.edge_classifier, MLPClassifier)
    assert curator.edge_classifier.activation == 'logistic'
    assert 'test_accuracy' in res_e
