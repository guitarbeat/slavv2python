import os
import pickle
import pytest
from slavv.analysis.ml_curator import MLCurator
# We expect pickle.UnpicklingError to be raised

class Malicious:
    def __reduce__(self):
        return (os.system, ("echo malicious code execution",))

def test_ml_curator_load_malicious_model(tmp_path):
    # Create malicious pickle
    filepath = tmp_path / "malicious_model.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(Malicious(), f)

    curator = MLCurator()

    # Attempt to load as vertex model
    # Should raise UnpicklingError because safe_load raises it and ml_curator doesn't catch it.

    with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle module"):
        curator.load_models(vertex_path=filepath)

    # Attempt to load as edge model
    with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle module"):
        curator.load_models(edge_path=filepath)
