"""
Unit tests for comparison metrics.

Tests the comparison functions from scripts/compare_matlab_python.py
for computing differences between MATLAB and Python results.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

from slavv.dev.metrics import (
    match_vertices,
    compare_vertices,
    compare_edges,
    compare_networks
)


class TestMatchVertices:
    """Tests for vertex matching algorithm."""
    
    def test_match_empty_arrays(self):
        """Test matching with empty arrays."""
        matlab_pos = np.array([])
        python_pos = np.array([])
        
        matlab_idx, python_idx, distances = match_vertices(matlab_pos, python_pos)
        
        assert len(matlab_idx) == 0
        assert len(python_idx) == 0
        assert len(distances) == 0
    
    def test_match_identical_positions(self):
        """Test matching identical vertex positions."""
        positions = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2],
            [20, 30, 40, 3]
        ])
        
        matlab_idx, python_idx, distances = match_vertices(positions, positions)
        
        assert len(matlab_idx) == 3
        assert len(python_idx) == 3
        assert np.allclose(distances, 0.0, atol=1e-10)
    
    def test_match_with_noise(self):
        """Test matching with small position differences."""
        matlab_pos = np.array([
            [10.0, 20.0, 30.0, 1],
            [15.0, 25.0, 35.0, 2]
        ])
        
        python_pos = np.array([
            [10.1, 20.1, 30.1, 1],  # 0.17 voxels away
            [15.2, 25.1, 35.1, 2]   # 0.24 voxels away
        ])
        
        matlab_idx, python_idx, distances = match_vertices(matlab_pos, python_pos)
        
        assert len(matlab_idx) == 2
        assert np.all(distances < 1.0)
    
    def test_match_with_threshold(self):
        """Test matching with distance threshold."""
        matlab_pos = np.array([
            [10, 20, 30, 1],
            [100, 200, 300, 2]  # Far away
        ])
        
        python_pos = np.array([
            [10.5, 20.5, 30.5, 1],  # Close match
            [50, 100, 150, 2]  # Different location
        ])
        
        # With default threshold (3.0)
        matlab_idx, python_idx, distances = match_vertices(
            matlab_pos, python_pos, distance_threshold=3.0
        )
        
        assert len(matlab_idx) == 1  # Only close match
        assert distances[0] < 1.0
    
    def test_match_more_matlab_than_python(self):
        """Test matching when MATLAB has more vertices."""
        matlab_pos = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2],
            [20, 30, 40, 3]
        ])
        
        python_pos = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2]
        ])
        
        matlab_idx, python_idx, distances = match_vertices(matlab_pos, python_pos)
        
        assert len(matlab_idx) <= len(matlab_pos)
        assert len(python_idx) <= len(python_pos)


class TestCompareVertices:
    """Tests for vertex comparison function."""
    
    def test_compare_empty_vertices(self):
        """Test comparing empty vertex data."""
        matlab_verts = {'count': 0, 'positions': np.array([]), 'radii': np.array([])}
        python_verts = {'count': 0, 'positions': np.array([]), 'radii': np.array([])}
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert result['matlab_count'] == 0
        assert result['python_count'] == 0
        assert result['count_difference'] == 0
    
    def test_compare_count_difference(self):
        """Test count difference calculation."""
        matlab_verts = {'count': 100, 'positions': np.array([]), 'radii': np.array([])}
        python_verts = {'count': 95, 'positions': np.array([]), 'radii': np.array([])}
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert result['count_difference'] == 5
        # Percent difference relative to average count
        expected_pct = (5 / 97.5) * 100
        assert np.isclose(result['count_percent_difference'], expected_pct)
    
    def test_compare_with_positions(self):
        """Test comparison including position matching."""
        positions = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2]
        ])
        
        matlab_verts = {
            'count': 2,
            'positions': positions,
            'radii': np.array([2.5, 3.0])
        }
        
        python_verts = {
            'count': 2,
            'positions': positions + 0.1,  # Slight offset
            'radii': np.array([2.5, 3.0])
        }
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert result['matched_vertices'] == 2
        assert result['position_rmse'] is not None
        assert result['position_rmse'] < 1.0
    
    def test_compare_with_radii(self):
        """Test comparison including radius correlation."""
        positions = np.array([[10, 20, 30, 1], [15, 25, 35, 2]])
        
        matlab_verts = {
            'count': 2,
            'positions': positions,
            'radii': np.array([2.5, 3.0])
        }
        
        python_verts = {
            'count': 2,
            'positions': positions,
            'radii': np.array([2.6, 3.1])  # Similar radii
        }
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert 'radius_correlation' in result
        assert result['radius_correlation'] is not None
        assert 'radius_stats' in result
    
    def test_compare_unmatched_vertices(self):
        """Test comparison with unmatched vertices."""
        matlab_pos = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2],
            [100, 100, 100, 3]  # Far away, won't match
        ])
        
        python_pos = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2]
        ])
        
        matlab_verts = {
            'count': 3,
            'positions': matlab_pos,
            'radii': np.array([2.5, 3.0, 3.5])
        }
        
        python_verts = {
            'count': 2,
            'positions': python_pos,
            'radii': np.array([2.5, 3.0])
        }
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert result['matched_vertices'] == 2
        assert result['unmatched_matlab'] == 1
        assert result['unmatched_python'] == 0


class TestCompareEdges:
    """Tests for edge comparison function."""
    
    def test_compare_empty_edges(self):
        """Test comparing empty edge data."""
        matlab_edges = {'count': 0, 'traces': [], 'total_length': 0.0}
        python_edges = {'count': 0, 'traces': []}
        
        result = compare_edges(matlab_edges, python_edges)
        
        assert result['matlab_count'] == 0
        assert result['python_count'] == 0
        assert result['count_difference'] == 0
    
    def test_compare_edge_counts(self):
        """Test edge count comparison."""
        matlab_edges = {'count': 50, 'traces': [], 'total_length': 0.0}
        python_edges = {'count': 48, 'traces': []}
        
        result = compare_edges(matlab_edges, python_edges)
        
        assert result['count_difference'] == 2
        expected_pct = (2 / 49.0) * 100
        assert np.isclose(result['count_percent_difference'], expected_pct)
    
    def test_compare_edge_lengths(self):
        """Test edge length comparison."""
        # Create mock edge traces
        trace1 = np.array([[0, 0, 0, 1], [1, 1, 1, 1], [2, 2, 2, 1]])
        trace2 = np.array([[0, 0, 0, 1], [3, 3, 3, 1]])
        
        matlab_edges = {
            'count': 2,
            'traces': [],
            'total_length': 100.0
        }
        
        python_edges = {
            'count': 2,
            'traces': [trace1, trace2]
        }
        
        result = compare_edges(matlab_edges, python_edges)
        
        assert 'total_length' in result
        assert 'matlab' in result['total_length']
        assert result['total_length']['matlab'] == 100.0


class TestCompareNetworks:
    """Tests for network comparison function."""
    
    def test_compare_empty_networks(self):
        """Test comparing empty network data."""
        matlab_stats = {'strand_count': 0}
        python_network = {'strands': []}
        
        result = compare_networks(matlab_stats, python_network)
        
        assert result['matlab_strand_count'] == 0
        assert result['python_strand_count'] == 0
    
    def test_compare_strand_counts(self):
        """Test strand count comparison."""
        matlab_stats = {'strand_count': 25}
        python_network = {'strands': [Mock() for _ in range(23)]}
        
        result = compare_networks(matlab_stats, python_network)
        
        assert result['matlab_strand_count'] == 25
        assert result['python_strand_count'] == 23
        assert result['strand_count_difference'] == 2
        expected_pct = (2 / 24.0) * 100
        assert np.isclose(result['strand_count_percent_difference'], expected_pct)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_match_vertices_single_point(self):
        """Test matching with single vertex."""
        matlab_pos = np.array([[10, 20, 30, 1]])
        python_pos = np.array([[10, 20, 30, 1]])
        
        matlab_idx, python_idx, distances = match_vertices(matlab_pos, python_pos)
        
        assert len(matlab_idx) == 1
        assert distances[0] < 1e-10
    
    def test_compare_vertices_no_matches(self):
        """Test comparison when no vertices match."""
        matlab_pos = np.array([[0, 0, 0, 1]])
        python_pos = np.array([[1000, 1000, 1000, 1]])
        
        matlab_verts = {
            'count': 1,
            'positions': matlab_pos,
            'radii': np.array([2.5])
        }
        
        python_verts = {
            'count': 1,
            'positions': python_pos,
            'radii': np.array([2.5])
        }
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert result['matched_vertices'] == 0
        assert result['unmatched_matlab'] == 1
        assert result['unmatched_python'] == 1
    
    def test_compare_with_nan_values(self):
        """Test handling of NaN values in data."""
        positions = np.array([[10, 20, 30, 1], [15, 25, 35, 2]])
        
        matlab_verts = {
            'count': 2,
            'positions': positions,
            'radii': np.array([2.5, np.nan])
        }
        
        python_verts = {
            'count': 2,
            'positions': positions,
            'radii': np.array([2.5, 3.0])
        }
        
        # Should handle gracefully without crashing
        result = compare_vertices(matlab_verts, python_verts)
        assert result is not None


class TestStatisticalMeasures:
    """Tests for statistical measures in comparisons."""
    
    def test_position_error_statistics(self):
        """Test computation of position error statistics."""
        # Create positions with known offset
        matlab_pos = np.array([
            [10, 20, 30, 1],
            [15, 25, 35, 2],
            [20, 30, 40, 3]
        ])
        
        offset = np.array([0.5, 0.5, 0.5, 0])
        python_pos = matlab_pos + offset
        
        matlab_verts = {
            'count': 3,
            'positions': matlab_pos,
            'radii': np.array([2.5, 3.0, 3.5])
        }
        
        python_verts = {
            'count': 3,
            'positions': python_pos,
            'radii': np.array([2.5, 3.0, 3.5])
        }
        
        result = compare_vertices(matlab_verts, python_verts)
        
        # Distance should be sqrt(0.5^2 * 3) ≈ 0.866
        expected_dist = np.sqrt(0.5**2 * 3)
        assert np.isclose(result['position_mean_distance'], expected_dist, rtol=0.01)
        assert np.isclose(result['position_rmse'], expected_dist, rtol=0.01)
    
    def test_radius_correlation_perfect(self):
        """Test radius correlation with perfect correlation."""
        positions = np.array([[10, 20, 30, 1], [15, 25, 35, 2]])
        radii = np.array([2.5, 3.0])
        
        matlab_verts = {
            'count': 2,
            'positions': positions,
            'radii': radii
        }
        
        python_verts = {
            'count': 2,
            'positions': positions,
            'radii': radii  # Identical
        }
        
        result = compare_vertices(matlab_verts, python_verts)
        
        assert result['radius_correlation'] is not None
        # Perfect correlation should have r ≈ 1.0
        assert np.isclose(result['radius_correlation']['pearson_r'], 1.0, atol=0.01)


# Fixtures

@pytest.fixture
def sample_matlab_vertices():
    """Create sample MATLAB vertex data."""
    return {
        'count': 10,
        'positions': np.random.rand(10, 4) * 100,
        'radii': np.random.rand(10) * 5 + 1.5
    }


@pytest.fixture
def sample_python_vertices(sample_matlab_vertices):
    """Create sample Python vertex data with small perturbations."""
    return {
        'count': 10,
        'positions': sample_matlab_vertices['positions'] + np.random.rand(10, 4) * 0.1,
        'radii': sample_matlab_vertices['radii'] + np.random.rand(10) * 0.1
    }


class TestWithFixtures:
    """Tests using fixtures for sample data."""
    
    def test_compare_sample_vertices(self, sample_matlab_vertices, sample_python_vertices):
        """Test comparison with sample data."""
        result = compare_vertices(sample_matlab_vertices, sample_python_vertices)
        
        assert result['matlab_count'] == 10
        assert result['python_count'] == 10
        assert result['matched_vertices'] > 0
        assert result['position_rmse'] is not None
    
    def test_compare_reproducibility(self, sample_matlab_vertices, sample_python_vertices):
        """Test that comparison results are reproducible."""
        result1 = compare_vertices(sample_matlab_vertices, sample_python_vertices)
        result2 = compare_vertices(sample_matlab_vertices, sample_python_vertices)
        
        assert result1['matched_vertices'] == result2['matched_vertices']
        assert result1['position_rmse'] == result2['position_rmse']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
