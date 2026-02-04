import pathlib
import sys
import numpy as np

# add src path
try:
    from slavv.ml_curator import extract_uncurated_info
except ImportError:
    from source.slavv.ml_curator import extract_uncurated_info


def test_extract_uncurated_info_shapes() -> None:
    vertices = {
        'positions': np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
        'energies': np.array([0.5, 0.6], dtype=float),
        'scales': np.array([1.0, 2.0], dtype=float),
        'radii_pixels': np.array([1.0, 1.0], dtype=float),
    }
    edges = {
        'traces': [np.array([[0, 0, 0], [1, 1, 1]], dtype=float)],
        'connections': np.array([[0, 1]], dtype=int),
        'energies': np.array([0.55], dtype=float),
    }
    energy = np.zeros((2, 2, 2), dtype=float)
    info = extract_uncurated_info(vertices, edges, {'energy': energy}, energy.shape)

    assert 'vertex_features' in info and 'edge_features' in info
    assert info['vertex_features'].shape[0] == 2
    assert info['edge_features'].shape[0] == 1
