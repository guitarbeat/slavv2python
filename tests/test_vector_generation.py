
import numpy as np
import pytest
from src.vectorization_core import SLAVVProcessor

processor = SLAVVProcessor()

def test_generate_edge_directions_returns_correct_shape():
    """Test that the function returns the correct number of vectors."""
    n_directions = 10
    directions = processor._generate_edge_directions(n_directions)
    assert directions.shape == (n_directions, 3)

def test_generate_edge_directions_returns_unit_vectors():
    """Test that all returned vectors are unit vectors."""
    n_directions = 10
    directions = processor._generate_edge_directions(n_directions)
    norms = np.linalg.norm(directions, axis=1)
    assert np.allclose(norms, 1.0)

def test_generate_edge_directions_handles_zero_directions():
    """Test that the function handles the case of zero directions."""
    directions = processor._generate_edge_directions(0)
    assert directions.shape == (0, 3)

def test_generate_edge_directions_handles_one_direction():
    """Test that the function handles the case of a single direction."""
    directions = processor._generate_edge_directions(1)
    assert directions.shape == (1, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.allclose(directions, [[0, 0, 1]])
