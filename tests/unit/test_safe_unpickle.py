import pytest
import os
import pickle
import joblib
import numpy as np
from slavv.utils.safe_unpickle import safe_load, SafeNumpyUnpickler, DEFAULT_MAX_PICKLE_SIZE

def test_safe_load_simple(tmp_path):
    data = {'a': 1, 'b': [1, 2, 3]}
    p = tmp_path / "test.pkl"
    with open(p, 'wb') as f:
        pickle.dump(data, f)

    loaded = safe_load(p)
    assert loaded == data

def test_safe_load_numpy(tmp_path):
    data = {'arr': np.array([1, 2, 3])}
    p = tmp_path / "test_numpy.pkl"
    with open(p, 'wb') as f:
        pickle.dump(data, f)

    loaded = safe_load(p)
    np.testing.assert_array_equal(loaded['arr'], data['arr'])

def test_safe_load_forbidden(tmp_path):
    # Create a malicious payload (simulated)
    # Since we can't easily create a malicious pickle without defining a class,
    # let's try to pickle an object from a forbidden module.
    import http.server

    class Forbidden:
        def __reduce__(self):
            return (http.server.SimpleHTTPRequestHandler, ())

    p = tmp_path / "forbidden.pkl"
    with open(p, 'wb') as f:
        pickle.dump(Forbidden(), f)

    with pytest.raises(pickle.UnpicklingError):
        safe_load(p)

def test_safe_load_size_limit(tmp_path):
    p = tmp_path / "large.pkl"
    with open(p, 'wb') as f:
        f.write(b'0' * (1024 + 1)) # Just dummy data

    # We can't pickle strictly here, but safe_load checks size first
    # So we don't need a valid pickle to test size check
    with pytest.raises(ValueError, match="exceeds limit"):
        safe_load(p, max_size=1024)

def test_safe_load_joblib(tmp_path):
    # Test compatibility with joblib dump (simple case)
    data = {'arr': np.array([1, 2, 3])}
    p = tmp_path / "test_joblib.pkl"
    joblib.dump(data, p)

    loaded = safe_load(p)
    np.testing.assert_array_equal(loaded['arr'], data['arr'])

def test_safe_load_missing_file():
    with pytest.raises(FileNotFoundError):
        safe_load("non_existent_file.pkl")
