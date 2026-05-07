import pickle

import numpy as np
import pytest
from slavv_python.utils.safe_unpickle import safe_load


def test_safe_load_simple(tmp_path):
    data = {"a": 1, "b": [1, 2, 3]}
    p = tmp_path / "test.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)

    loaded = safe_load(p)
    assert loaded == data


def test_safe_load_numpy(tmp_path):
    data = {"arr": np.array([1, 2, 3])}
    p = tmp_path / "test_numpy.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)

    loaded = safe_load(p)
    np.testing.assert_array_equal(loaded["arr"], data["arr"])


def test_safe_load_forbidden(tmp_path):
    # Create a malicious payload (simulated)
    # Since we can't easily create a malicious pickle without defining a class,
    # let's try to pickle an object from a forbidden module.
    import http.server

    class Forbidden:
        def __reduce__(self):
            return (http.server.SimpleHTTPRequestHandler, ())

    p = tmp_path / "forbidden.pkl"
    with open(p, "wb") as f:
        pickle.dump(Forbidden(), f)

    with pytest.raises(pickle.UnpicklingError):
        safe_load(p)


def test_safe_load_rejects_builtin_eval_gadget(tmp_path):
    class Forbidden:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    p = tmp_path / "builtin_eval.pkl"
    with open(p, "wb") as f:
        pickle.dump(Forbidden(), f)

    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        safe_load(p)
