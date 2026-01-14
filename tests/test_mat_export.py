import pathlib
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.slavv.visualization import NetworkVisualizer
from src.slavv.io_utils import load_network_from_mat


def test_export_and_load_mat(tmp_path: Path) -> None:
    vertices = {
        'positions': np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
        'energies': np.array([-1.0, -2.0], dtype=float),
        'radii_microns': np.array([1.0, 1.5], dtype=float),
        'radii_pixels': np.array([2.0, 3.0], dtype=float),
        'scales': np.array([0.5, 0.5], dtype=float),
    }
    edges = {
        'connections': np.array([[0, 1]], dtype=int),
        'traces': [np.array([[0, 0, 0], [1, 1, 1]], dtype=float)],
    }
    network = {
        'strands': [np.array([0])],
        'bifurcations': np.array([], dtype=int),
        'vertex_degrees': np.array([1, 1], dtype=int),
    }
    params = {'voxel_spacing': [1.0, 1.0, 1.0]}
    processing_results = {
        'vertices': vertices,
        'edges': edges,
        'network': network,
        'parameters': params,
    }

    out_path = tmp_path / 'network.mat'
    NetworkVisualizer().export_network_data(processing_results, out_path, format='mat')

    loaded = load_network_from_mat(out_path)
    assert np.array_equal(loaded.vertices, vertices['positions'])
    assert np.array_equal(loaded.edges, edges['connections'])
    assert np.array_equal(loaded.radii, vertices['radii_microns'])
