import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src')
)

from vectorization_core import SLAVVProcessor


def test_extract_edges_watershed_two_vertices():
    processor = SLAVVProcessor()
    energy = np.ones((5, 5, 5), dtype=np.float32)
    energy_data = {
        'energy': energy,
        'scale_indices': np.zeros_like(energy, dtype=np.int16),
        'lumen_radius_pixels': np.array([1.0], dtype=np.float32),
        'lumen_radius_microns': np.array([1.0], dtype=np.float32),
        'energy_sign': -1.0,
    }
    vertices = {
        'positions': np.array([[0, 0, 0], [4, 4, 4]], dtype=float),
    }
    edges = processor.extract_edges_watershed(energy_data, vertices, {})
    assert edges['connections'].shape == (1, 2)
    assert edges['connections'][0].tolist() == [0, 1]
    assert len(edges['traces']) == 1
    assert edges['traces'][0].shape[1] == 3
    assert np.isclose(edges['energies'][0], 1.0)
