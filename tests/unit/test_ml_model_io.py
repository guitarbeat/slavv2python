import pathlib
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from slavv.analysis import MLCurator


def test_save_and_load_models(tmp_path):
    curator = MLCurator()

    # Create and fit simple vertex classifier
    Xv = np.random.rand(20, 3)
    yv = np.random.randint(0, 2, 20)
    curator.vertex_scaler.fit(Xv)
    curator.vertex_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.vertex_classifier.fit(curator.vertex_scaler.transform(Xv), yv)

    # Create and fit simple edge classifier
    Xe = np.random.rand(20, 3)
    ye = np.random.randint(0, 2, 20)
    curator.edge_scaler.fit(Xe)
    curator.edge_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.edge_classifier.fit(curator.edge_scaler.transform(Xe), ye)

    vertex_path = tmp_path / "vertex.joblib"
    edge_path = tmp_path / "edge.joblib"
    curator.save_models(vertex_path, edge_path)

    loaded = MLCurator()
    loaded.load_models(vertex_path, edge_path)

    X_test_v = np.random.rand(5, 3)
    orig_v = curator.vertex_classifier.predict(curator.vertex_scaler.transform(X_test_v))
    new_v = loaded.vertex_classifier.predict(loaded.vertex_scaler.transform(X_test_v))
    assert np.array_equal(orig_v, new_v)

    X_test_e = np.random.rand(5, 3)
    orig_e = curator.edge_classifier.predict(curator.edge_scaler.transform(X_test_e))
    new_e = loaded.edge_classifier.predict(loaded.edge_scaler.transform(X_test_e))
    assert np.array_equal(orig_e, new_e)
