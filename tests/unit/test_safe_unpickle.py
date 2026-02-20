import os
import pickle
import pytest
import numpy as np
from slavv.utils.safe_unpickle import safe_load, SafeNumpyUnpickler

class Malicious:
    def __reduce__(self):
        return (os.system, ("echo malicious code execution",))

def test_safe_load_valid(tmp_path):
    # Test valid numpy array
    data = np.array([1, 2, 3])
    filepath = tmp_path / "valid.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    loaded = safe_load(filepath)
    np.testing.assert_array_equal(loaded, data)

def test_safe_load_dict(tmp_path):
    # Test valid dict
    data = {"a": 1, "b": 2}
    filepath = tmp_path / "dict.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    loaded = safe_load(filepath)
    assert loaded == data

def test_safe_load_malicious(tmp_path):
    # Test malicious pickle
    filepath = tmp_path / "malicious.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(Malicious(), f)

    with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle module"):
        safe_load(filepath)

def test_safe_load_compressed(tmp_path):
    # Test compressed file (gzip)
    import gzip
    data = {"compressed": True}
    filepath = tmp_path / "compressed.pkl.gz"
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(data, f)

    loaded = safe_load(filepath)
    assert loaded == data
