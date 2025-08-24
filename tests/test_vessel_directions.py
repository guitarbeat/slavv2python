import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src')
)

from vectorization_core import SLAVVProcessor


def test_estimate_vessel_directions_axis_aligned():
    processor = SLAVVProcessor()
    coords = np.indices((21, 21, 21))
    y = coords[0] - 10
    z = coords[2] - 10
    energy = np.exp(-(y**2 + z**2) / (2 * 2**2))
    pos = np.array([10, 10, 10], dtype=float)
    dirs = processor._estimate_vessel_directions(energy, pos, radius=4.0)
    assert dirs.shape == (2, 3)
    assert np.allclose(np.abs(dirs[0]), np.array([0, 1, 0]), atol=0.2)


def test_estimate_vessel_directions_fallback():
    processor = SLAVVProcessor()
    energy = np.zeros((2, 2, 2), dtype=float)
    pos = np.zeros(3)
    dirs = processor._estimate_vessel_directions(energy, pos, radius=0.5)
    expected = processor._generate_edge_directions(2)
    assert np.allclose(dirs, expected)

