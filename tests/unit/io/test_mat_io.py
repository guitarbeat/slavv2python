"""Consolidated tests for MAT file I/O."""
import numpy as np
from pathlib import Path
from scipy.io import savemat

from slavv.io import load_network_from_mat
from slavv.visualization import NetworkVisualizer


def test_mat_roundtrip(tmp_path: Path) -> None:
    """Test loading a MAT file with network data."""
    vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    edges = np.array([[0, 1]], dtype=int)
    radii = np.array([1.0, 2.0], dtype=float)
    mat_path = tmp_path / "network.mat"
    savemat(mat_path, {"vertices": vertices, "edges": edges, "radii": radii})

    network = load_network_from_mat(mat_path)

    assert np.array_equal(network.vertices, vertices)
    assert np.array_equal(network.edges, edges)
    assert np.array_equal(network.radii, radii)


def test_mat_export_via_visualizer(tmp_path: Path) -> None:
    """Test exporting network via NetworkVisualizer and reloading."""
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
    processing_results = {
        'vertices': vertices,
        'edges': edges,
        'network': network,
        'parameters': {'voxel_spacing': [1.0, 1.0, 1.0]},
    }

    out_path = tmp_path / 'network.mat'
    NetworkVisualizer().export_network_data(processing_results, out_path, format='mat')

    loaded = load_network_from_mat(out_path)
    assert np.array_equal(loaded.vertices, vertices['positions'])
    assert np.array_equal(loaded.edges, edges['connections'])
    assert np.array_equal(loaded.radii, vertices['radii_microns'])


def test_mat_export_complex_params(tmp_path: Path) -> None:
    """Test exporting parameters with mixed types (Bug 13)."""
    vertices = {'positions': [], 'energies': [], 'scales': []}
    edges = {'connections': [], 'traces': []}
    network = {}
    
    # Complex parameters that would fail without sanitization
    parameters = {
        'voxel_spacing': [1.0, 1.0, 1.0],
        'metadata': {
            'date': '2023-01-01',
            'ignore_this': None,
            'settings': {'threshold': 0.5}
        },
        'flags': {True, False}, # Set (not JSON/MAT serializable usually)
        'none_value': None,
    }
    
    processing_results = {
        'vertices': vertices,
        'edges': edges,
        'network': network,
        'parameters': parameters,
    }

    out_path = tmp_path / 'complex.mat'
    # Should not raise error
    NetworkVisualizer().export_network_data(processing_results, out_path, format='mat')
    
    assert out_path.exists()
