import pytest
import os
import pickle
import joblib
import numpy as np
import gzip
from sklearn.ensemble import RandomForestClassifier
from slavv.utils import safe_load

def test_safe_load_valid_model(tmp_path):
    """Test that safe_load can load a valid sklearn model."""
    # Create a dummy model
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)

    # Save model
    model_path = tmp_path / "model.joblib"
    joblib.dump(clf, model_path)

    # Load model with safe_load
    loaded_clf = safe_load(str(model_path))

    # Verify it works
    assert isinstance(loaded_clf, RandomForestClassifier)
    assert np.allclose(loaded_clf.predict([[0, 0]]), [0])

def test_safe_load_compressed_model(tmp_path):
    """Test that safe_load can load a compressed model."""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)

    # Save compressed model
    model_path = tmp_path / "model.joblib.gz"
    # joblib automatically handles compression based on extension
    joblib.dump(clf, model_path)

    # Load model with safe_load
    loaded_clf = safe_load(str(model_path))

    assert isinstance(loaded_clf, RandomForestClassifier)

def test_safe_load_blocks_malicious_pickle(tmp_path):
    """Test that safe_load blocks malicious pickle files."""
    class ForbiddenClass:
        def __reduce__(self):
            return (os.system, ("echo malicious code executed",))

    malicious_data = pickle.dumps(ForbiddenClass())
    malicious_path = tmp_path / "malicious.pkl"
    with open(malicious_path, 'wb') as f:
        f.write(malicious_data)

    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        safe_load(str(malicious_path))

def test_safe_load_blocks_unsafe_builtins(tmp_path):
    """Test that safe_load blocks unsafe builtins like eval."""
    # Pickle a payload using builtins.eval (conceptually)
    # We can't easily pickle 'eval' directly as a callable unless we use cloudpickle or similar trickery,
    # but we can try to pickle something that references it.

    # Simpler: Create a pickle that references 'builtins.eval'
    # We can craft it manually or use a class that reduces to eval.

    class MaliciousEval:
        def __reduce__(self):
            return (eval, ("print('pwned')",))

    malicious_data = pickle.dumps(MaliciousEval())
    malicious_path = tmp_path / "eval.pkl"
    with open(malicious_path, 'wb') as f:
        f.write(malicious_data)

    with pytest.raises(pickle.UnpicklingError, match="Global 'builtins.eval' is forbidden"):
        safe_load(str(malicious_path))
