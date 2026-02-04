"""
Comparison metrics for SLAVV validation.

This module contains functions to compare vertices, edges, and network statistics
between MATLAB and Python implementations.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

def match_vertices(
    matlab_positions: np.ndarray,
    python_positions: np.ndarray,
    distance_threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match vertices between MATLAB and Python using nearest neighbors."""
    if matlab_positions.size == 0 or python_positions.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Use only spatial coordinates (first 3 columns)
    matlab_xyz = matlab_positions[:, :3] if matlab_positions.ndim == 2 else matlab_positions.reshape(-1, 3)
    python_xyz = python_positions[:, :3] if python_positions.ndim == 2 else python_positions.reshape(-1, 3)
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(python_xyz)
    
    # Find nearest Python vertex for each MATLAB vertex
    distances, python_indices = tree.query(matlab_xyz)
    
    # Filter by distance threshold
    valid_matches = distances < distance_threshold
    
    matlab_indices = np.arange(len(matlab_xyz))[valid_matches]
    matched_python_indices = python_indices[valid_matches]
    matched_distances = distances[valid_matches]
    
    return matlab_indices, matched_python_indices, matched_distances


def compare_vertices(matlab_verts: Dict[str, Any], python_verts: Dict[str, Any]) -> Dict[str, Any]:
    """Compare vertex information between MATLAB and Python."""
    comparison = {
        'matlab_count': matlab_verts.get('count', 0),
        'python_count': python_verts.get('count', 0),
        'count_difference': 0,
        'count_percent_difference': 0.0,
        'position_rmse': None,
        'matched_vertices': 0,
        'unmatched_matlab': 0,
        'unmatched_python': 0,
        'radius_correlation': None,
        'radius_stats': {}
    }
    
    matlab_count = comparison['matlab_count']
    python_count = comparison['python_count']
    
    if matlab_count > 0 or python_count > 0:
        comparison['count_difference'] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison['count_percent_difference'] = (comparison['count_difference'] / avg_count) * 100.0
    
    # Match vertices if both have data
    matlab_positions = matlab_verts.get('positions', np.array([]))
    python_positions = python_verts.get('positions', np.array([]))
    
    if matlab_positions.size > 0 and python_positions.size > 0:
        matlab_idx, python_idx, distances = match_vertices(matlab_positions, python_positions)
        
        comparison['matched_vertices'] = len(matlab_idx)
        comparison['unmatched_matlab'] = matlab_count - len(matlab_idx)
        comparison['unmatched_python'] = python_count - len(python_idx)
        
        if len(distances) > 0:
            comparison['position_rmse'] = float(np.sqrt(np.mean(distances**2)))
            comparison['position_mean_distance'] = float(np.mean(distances))
            comparison['position_median_distance'] = float(np.median(distances))
            comparison['position_95th_percentile'] = float(np.percentile(distances, 95))
        
        # Compare radii for matched vertices
        matlab_radii = matlab_verts.get('radii', np.array([]))
        python_radii = python_verts.get('radii', np.array([]))
        
        if len(matlab_idx) > 0 and matlab_radii.size > 0 and python_radii.size > 0:
            matched_matlab_radii = matlab_radii[matlab_idx]
            matched_python_radii = python_radii[python_idx]
            
            # Compute correlation
            if len(matched_matlab_radii) > 1:
                pearson_r, pearson_p = stats.pearsonr(matched_matlab_radii, matched_python_radii)
                spearman_r, spearman_p = stats.spearmanr(matched_matlab_radii, matched_python_radii)
                
                comparison['radius_correlation'] = {
                    'pearson_r': float(pearson_r),
                    'pearson_p': float(pearson_p),
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p)
                }
            
            # Radius statistics
            comparison['radius_stats'] = {
                'matlab_mean': float(np.mean(matched_matlab_radii)),
                'matlab_std': float(np.std(matched_matlab_radii)),
                'python_mean': float(np.mean(matched_python_radii)),
                'python_std': float(np.std(matched_python_radii)),
                'mean_difference': float(np.mean(matched_matlab_radii - matched_python_radii)),
                'rmse': float(np.sqrt(np.mean((matched_matlab_radii - matched_python_radii)**2)))
            }
    
    return comparison


def compare_edges(matlab_edges: Dict[str, Any], python_edges: Dict[str, Any]) -> Dict[str, Any]:
    """Compare edge information between MATLAB and Python."""
    comparison = {
        'matlab_count': matlab_edges.get('count', 0),
        'python_count': python_edges.get('count', 0),
        'count_difference': 0,
        'count_percent_difference': 0.0,
        'total_length': {}
    }
    
    matlab_count = comparison['matlab_count']
    python_count = comparison['python_count']
    
    if matlab_count > 0 or python_count > 0:
        comparison['count_difference'] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison['count_percent_difference'] = (comparison['count_difference'] / avg_count) * 100.0
    
    # Compare total lengths if available
    matlab_total_length = matlab_edges.get('total_length', 0.0)
    if matlab_total_length > 0:
        comparison['total_length']['matlab'] = float(matlab_total_length)
    
    # Calculate Python edge lengths
    python_traces = python_edges.get('traces', [])
    if python_traces:
        python_total_length = 0.0
        for trace in python_traces:
            if isinstance(trace, np.ndarray) and trace.size > 0:
                # Calculate path length
                if trace.ndim == 2 and trace.shape[0] > 1:
                    diffs = np.diff(trace[:, :3], axis=0)
                    lengths = np.sqrt(np.sum(diffs**2, axis=1))
                    python_total_length += np.sum(lengths)
        
        comparison['total_length']['python'] = float(python_total_length)
        
        if matlab_total_length > 0 and python_total_length > 0:
            comparison['total_length']['difference'] = float(abs(matlab_total_length - python_total_length))
            comparison['total_length']['percent_difference'] = float(
                (comparison['total_length']['difference'] / ((matlab_total_length + python_total_length) / 2.0)) * 100.0
            )
    
    return comparison


def compare_networks(matlab_stats: Dict[str, Any], python_network: Dict[str, Any]) -> Dict[str, Any]:
    """Compare network-level statistics."""
    comparison = {
        'matlab_strand_count': matlab_stats.get('strand_count', 0),
        'python_strand_count': 0
    }
    
    # Extract Python strand count
    if 'strands' in python_network:
        comparison['python_strand_count'] = len(python_network['strands'])
    
    matlab_count = comparison['matlab_strand_count']
    python_count = comparison['python_strand_count']
    
    if matlab_count > 0 or python_count > 0:
        comparison['strand_count_difference'] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison['strand_count_percent_difference'] = (comparison['strand_count_difference'] / avg_count) * 100.0
    
    return comparison
