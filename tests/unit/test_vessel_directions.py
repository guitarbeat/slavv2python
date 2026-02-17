import pathlib
import sys
import numpy as np
from unittest.mock import patch

# Add source path for imports
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src')
)

from slavv.core import SLAVVProcessor


def test_estimate_vessel_directions_axis_aligned():
    processor = SLAVVProcessor()
    coords = np.indices((21, 21, 21))
    y = coords[0] - 10
    z = coords[2] - 10
    energy = np.exp(-(y**2 + z**2) / (2 * 2**2))
    pos = np.array([10, 10, 10], dtype=float)
    dirs = processor._estimate_vessel_directions(
        energy, pos, radius=4.0, microns_per_voxel=np.array([1.0, 1.0, 1.0])
    )
    assert dirs.shape == (2, 3)
    assert np.allclose(np.abs(dirs[0]), np.array([0, 1, 0]), atol=0.2)


@patch(
    'slavv.core.tracing.generate_edge_directions',
    return_value=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float),
)
def test_estimate_vessel_directions_fallback(mock_generate_directions):
    processor = SLAVVProcessor()
    energy = np.zeros((2, 2, 2), dtype=float)
    pos = np.zeros(3)
    dirs = processor._estimate_vessel_directions(
        energy, pos, radius=0.5, microns_per_voxel=np.array([1.0, 1.0, 1.0])
    )
    expected = processor._generate_edge_directions(2)
    assert np.allclose(dirs, expected)


def test_estimate_vessel_directions_anisotropic_spacing():
    processor = SLAVVProcessor()
    coords = np.indices((21, 21, 21))
    x = coords[1] - 10
    z = (coords[2] - 10) * 2  # simulate stretched z axis
    energy = np.exp(-(x**2 + z**2) / (2 * 2**2))
    pos = np.array([10, 10, 10], dtype=float)
    dirs = processor._estimate_vessel_directions(
        energy, pos, radius=4.0, microns_per_voxel=np.array([1.0, 1.0, 2.0])
    )
    assert dirs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-6)
    assert np.allclose(dirs[0], -dirs[1])


@patch(
    'slavv.core.tracing.generate_edge_directions',
    return_value=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float),
)
def test_estimate_vessel_directions_isotropic_hessian(mock_generate_directions):
    processor = SLAVVProcessor()
    energy = np.ones((21, 21, 21), dtype=float)
    pos = np.array([10, 10, 10], dtype=float)
    dirs = processor._estimate_vessel_directions(
        energy, pos, radius=4.0, microns_per_voxel=np.array([1.0, 1.0, 1.0])
    )
    expected = processor._generate_edge_directions(2)
    assert np.allclose(dirs, expected)

