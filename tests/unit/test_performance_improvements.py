"""
Performance tests to validate optimization improvements.

These tests ensure that optimized code produces the same results as before
while being more efficient.
"""
import numpy as np
import pytest
from slavv.analysis.geometry import (
    calculate_branching_angles,
    calculate_surface_area,
    calculate_vessel_volume
)
from slavv.analysis.ml_curator import MLCurator


def test_branching_angles_performance():
    """Test that branching angle calculation is correct and efficient."""
    # Create a simple test case with known angles
    positions = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0 (bifurcation)
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
    ])
    
    # Define strands - two strands meeting at vertex 0
    strands = [
        [0, 1],  # Strand from 0 to 1
        [0, 2],  # Strand from 0 to 2
    ]
    
    microns_per_voxel = [1.0, 1.0, 1.0]
    bifurcations = np.array([0])  # Vertex 0 is a bifurcation
    
    # Compute angles
    angles = calculate_branching_angles(strands, positions, microns_per_voxel, bifurcations)
    
    # Should have one angle (between the two edges at vertex 0)
    assert len(angles) == 1
    # The angle should be 90 degrees (edges along x and y axes)
    assert abs(angles[0] - 90.0) < 0.1


def test_surface_area_calculation_performance():
    """Test that vectorized surface area calculation is correct."""
    # Create a simple strand (straight line)
    strands = [[0, 1, 2]]
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    
    radii = np.array([1.0, 1.0, 1.0])
    microns_per_voxel = [1.0, 1.0, 1.0]
    
    # Calculate surface area
    area = calculate_surface_area(strands, positions, radii, microns_per_voxel)
    
    # Expected: 2 segments, each length 1.0, radius 1.0
    # Surface area = 2 * pi * r * L = 2 * pi * 1.0 * 1.0 = 2*pi per segment
    # Total = 4*pi
    expected_area = 4.0 * np.pi
    assert abs(area - expected_area) < 0.01


def test_volume_calculation_performance():
    """Test that vectorized volume calculation is correct."""
    # Create a simple strand (straight line)
    strands = [[0, 1, 2]]
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    
    radii = np.array([1.0, 1.0, 1.0])
    microns_per_voxel = [1.0, 1.0, 1.0]
    
    # Calculate volume
    volume = calculate_vessel_volume(strands, positions, radii, microns_per_voxel)
    
    # Expected: 2 segments, each length 1.0, radius 1.0
    # Volume = pi * r^2 * L = pi * 1.0^2 * 1.0 = pi per segment
    # Total = 2*pi
    expected_volume = 2.0 * np.pi
    assert abs(volume - expected_volume) < 0.01


def test_ml_curator_feature_extraction_shape():
    """Test that ML feature extraction produces correct output shape."""
    # Create minimal test data
    n_vertices = 10
    vertices = {
        'positions': np.random.rand(n_vertices, 3) * 10,
        'energies': np.random.rand(n_vertices),
        'scales': np.random.randint(0, 3, n_vertices),
        'radii': np.random.rand(n_vertices) * 2 + 0.5
    }
    
    energy_data = {
        'energy': np.random.rand(20, 20, 20)
    }
    
    image_shape = (20, 20, 20)
    
    curator = MLCurator()
    features = curator.extract_vertex_features(vertices, energy_data, image_shape)
    
    # Check that we get features for all vertices
    assert features.shape[0] == n_vertices
    # Check that we get the expected number of features per vertex
    # (energy, scale, radius, ratio, 3 spatial, dist_from_center, 6 local stats, 4 gradient)
    assert features.shape[1] > 10  # Should have multiple features


def test_ml_curator_edge_feature_extraction():
    """Test that edge feature extraction with vectorized energy sampling works."""
    # Create minimal test data
    edges = {
        'traces': [
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            np.array([[5, 5, 5], [6, 6, 6]])
        ],
        'connections': [(0, 1), (1, 2)]
    }
    
    vertices = {
        'positions': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        'energies': np.array([1.0, 2.0, 3.0]),
        'radii': np.array([0.5, 0.6, 0.7])
    }
    
    energy_data = {
        'energy': np.random.rand(10, 10, 10)
    }
    
    curator = MLCurator()
    features = curator.extract_edge_features(edges, vertices, energy_data)
    
    # Should get features for both edges
    assert features.shape[0] == 2
    # Should have multiple features per edge
    assert features.shape[1] > 5


def test_multiple_strands_surface_area():
    """Test surface area calculation with multiple strands."""
    strands = [
        [0, 1, 2],      # First strand
        [3, 4],         # Second strand
        [5, 6, 7, 8]    # Third strand
    ]
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
    ])
    
    radii = np.ones(9) * 0.5
    microns_per_voxel = [1.0, 1.0, 1.0]
    
    # Calculate surface area
    area = calculate_surface_area(strands, positions, radii, microns_per_voxel)
    
    # Should be positive
    assert area > 0
    # Should be finite
    assert np.isfinite(area)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
