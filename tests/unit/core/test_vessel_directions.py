from unittest.mock import patch

import numpy as np
import pytest

from slavv_python.core.edge_primitives import estimate_vessel_directions, generate_edge_directions


def test_estimate_vessel_directions_axis_aligned():
    coords = np.indices((21, 21, 21))
    y = coords[0] - 10
    z = coords[2] - 10
    energy = np.exp(-(y**2 + z**2) / (2 * 2**2))
    pos = np.array([10, 10, 10], dtype=float)
    dirs = estimate_vessel_directions(
        energy, pos, radius=4.0, microns_per_voxel=np.array([1.0, 1.0, 1.0])
    )
    assert dirs.shape == (2, 3)
    assert np.allclose(np.abs(dirs[0]), np.array([0, 1, 0]), atol=0.2)


@patch(
    "slavv_python.core.edge_primitives.generate_edge_directions",
    return_value=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float),
)
def test_estimate_vessel_directions_fallback(mock_generate_directions):
    energy = np.zeros((2, 2, 2), dtype=float)
    pos = np.zeros(3)
    dirs = estimate_vessel_directions(
        energy, pos, radius=0.5, microns_per_voxel=np.array([1.0, 1.0, 1.0])
    )
    expected = mock_generate_directions.return_value
    assert np.allclose(dirs, expected)


def test_estimate_vessel_directions_anisotropic_spacing():
    coords = np.indices((21, 21, 21))
    x = coords[1] - 10
    z = (coords[2] - 10) * 2  # simulate stretched z axis
    energy = np.exp(-(x**2 + z**2) / (2 * 2**2))
    pos = np.array([10, 10, 10], dtype=float)
    dirs = estimate_vessel_directions(
        energy, pos, radius=4.0, microns_per_voxel=np.array([1.0, 1.0, 2.0])
    )
    assert dirs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-6)
    assert np.allclose(dirs[0], -dirs[1])


@patch(
    "slavv_python.core.edge_primitives.generate_edge_directions",
    return_value=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float),
)
def test_estimate_vessel_directions_isotropic_hessian(mock_generate_directions):
    energy = np.ones((21, 21, 21), dtype=float)
    pos = np.array([10, 10, 10], dtype=float)
    dirs = estimate_vessel_directions(
        energy, pos, radius=4.0, microns_per_voxel=np.array([1.0, 1.0, 1.0])
    )
    expected = mock_generate_directions.return_value
    assert np.allclose(dirs, expected)


@pytest.mark.parametrize(
    ("n_directions", "expected_shape"),
    [
        (0, (0, 3)),
        (1, (1, 3)),
        (5, (5, 3)),
        (10, (10, 3)),
    ],
)
def test_generate_edge_directions_shape(n_directions, expected_shape):
    """Test that the function returns correct shape for various inputs."""
    directions = generate_edge_directions(n_directions)
    assert directions.shape == expected_shape


def test_generate_edge_directions_properties():
    """Test that returned vectors are unit vectors and unique."""
    # Single direction should return [0, 0, 1]
    dirs1 = generate_edge_directions(1)
    assert np.allclose(dirs1[0], [0, 0, 1])

    # Multiple directions should be unit vectors
    dirs5 = generate_edge_directions(5)
    norms = np.linalg.norm(dirs5, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Directions should be unique
    assert len(np.unique(np.round(dirs5, 6), axis=0)) == 5
