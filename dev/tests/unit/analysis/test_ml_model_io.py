import numpy as np
from sklearn.ensemble import RandomForestClassifier

from slavv.analysis import MLCurator


class _UploadedModel:
    def __init__(self, path):
        self.name = path.name
        self._payload = path.read_bytes()

    def getvalue(self):
        return self._payload


def test_save_and_load_models(tmp_path):
    curator = MLCurator()
    rng = np.random.default_rng(0)

    # Create and fit simple vertex classifier
    Xv = rng.random((20, 3))
    yv = rng.integers(0, 2, size=20)
    curator.vertex_scaler.fit(Xv)
    curator.vertex_classifier = RandomForestClassifier(n_estimators=1, random_state=0)
    curator.vertex_classifier.fit(curator.vertex_scaler.transform(Xv), yv)

    # Create and fit simple edge classifier
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


def test_load_models_accepts_uploaded_file_objects(tmp_path):
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
