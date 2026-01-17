"""Consolidated tests for generate_edge_directions function."""
import numpy as np
import pytest
from src.slavv.tracing import generate_edge_directions


@pytest.mark.parametrize("n_directions,expected_shape", [
    (0, (0, 3)),
    (1, (1, 3)),
    (5, (5, 3)),
    (10, (10, 3)),
])
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
