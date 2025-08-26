import pathlib
import sys
import numpy as np
from pathlib import Path
from scipy.io import savemat

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from io_utils import load_network_from_mat


def test_load_network_from_mat(tmp_path: Path) -> None:
    vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    edges = np.array([[0, 1]], dtype=int)
    radii = np.array([1.0, 2.0], dtype=float)
    mat_path = tmp_path / "network.mat"
    savemat(mat_path, {"vertices": vertices, "edges": edges, "radii": radii})

    network = load_network_from_mat(mat_path)

    assert np.array_equal(network.vertices, vertices)
    assert np.array_equal(network.edges, edges)
    assert np.array_equal(network.radii, radii)
