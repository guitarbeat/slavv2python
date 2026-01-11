import pathlib
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from io_utils import (
    Network,
    load_network_from_csv,
    load_network_from_json,
    save_network_to_csv,
    save_network_to_json,
)


def test_save_and_load_csv(tmp_path: Path) -> None:
    network = Network(
        vertices=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        edges=np.array([[0, 1]], dtype=int),
        radii=np.array([4.0, 7.0], dtype=float),
    )

    v_path, e_path = save_network_to_csv(network, tmp_path / 'net')
    assert v_path.exists() and e_path.exists()

    loaded = load_network_from_csv(tmp_path / 'net')
    assert np.allclose(loaded.vertices, network.vertices)
    assert np.array_equal(loaded.edges, network.edges)
    assert np.allclose(loaded.radii, network.radii)


def test_save_and_load_json(tmp_path: Path) -> None:
    network = Network(
        vertices=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        edges=np.array([[0, 1]], dtype=int),
        radii=np.array([4.0, 7.0], dtype=float),
    )

    path = save_network_to_json(network, tmp_path / 'net.json')
    assert Path(path).exists()

    loaded = load_network_from_json(path)
    assert np.allclose(loaded.vertices, network.vertices)
    assert np.array_equal(loaded.edges, network.edges)
    assert np.allclose(loaded.radii, network.radii)
