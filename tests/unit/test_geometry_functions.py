import numpy as np
import pytest
<<<<<<< HEAD
from slavv.analysis import (
=======
from slavv.analysis.geometry import (
>>>>>>> 02551966425602193b36f418552db1552ddb39ea
    resample_vectors,
    smooth_edge_traces,
    transform_vector_set,
    register_vector_sets,
    get_edges_for_vertex
)

def test_resample_vectors_straight_line():
    # Straight line from (0,0) to (10,0)
    trace = np.array([[0.0, 0.0], [10.0, 0.0]])
    step = 1.0
    resampled = resample_vectors(trace, step)
    
    # Expect 11 points: 0, 1, 2, ..., 10
    assert len(resampled) == 11
    expected = np.column_stack((np.linspace(0, 10, 11), np.zeros(11)))
    np.testing.assert_allclose(resampled, expected, atol=1e-6)

def test_resample_vectors_curved():
    # Corner: (0,0) -> (5,0) -> (5,5)
    # Total length = 10
    trace = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
    step = 2.0
    resampled = resample_vectors(trace, step)
    
    # Total length 10, step 2 -> 0, 2, 4, 6, 8, 10 (6 points)
    assert len(resampled) == 6
    # Check start and end
    np.testing.assert_allclose(resampled[0], [0, 0], atol=1e-6)
    np.testing.assert_allclose(resampled[-1], [5, 5], atol=1e-6)
    # Check midpoint (approximate arc length 5) -> (5,0)
    # Note: resample is linear interpolation, so it cuts corners slightly less than perfect arc
    # But for this simple polyline, points lie exactly on the segments
    np.testing.assert_allclose(resampled[2], [4, 0], atol=1e-6) # dist 4
    np.testing.assert_allclose(resampled[3], [5, 1], atol=1e-6) # dist 6 (5+1)

def test_smooth_edge_traces():
    # Zig-zag pattern
    trace = np.array([[0,0], [1,1], [2,0], [3,1], [4,0]], dtype=float)
    traces = [trace]
    smoothed = smooth_edge_traces(traces, sigma=1.0)
    
    assert len(smoothed) == 1
    s_trace = smoothed[0]
    
    # Endpoints should settle somewhat but smoothing pulls them in? 
    # scipy.ndimage.gaussian_filter1d with mode='nearest' preserves mean of flat signals
    # but smooths peaks.
    
    # The peaks (1,1) and (3,1) should be lowered
    assert s_trace[1,1] < 1.0
    assert s_trace[3,1] < 1.0
    
    # The valleys (2,0) should be raised
    assert s_trace[2,1] > 0.0

def test_transform_vector_set_scale_translate():
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    
    # Scale x2, Translate +1
    scale = [2.0, 2.0, 2.0]
    translate = [1.0, 1.0, 1.0]
    
    transformed = transform_vector_set(points, scale=scale, translate=translate)
    
    expected = np.array([
        [3, 1, 1],
        [1, 3, 1],
        [1, 1, 3]
    ], dtype=float)
    
    np.testing.assert_allclose(transformed, expected, atol=1e-6)

def test_register_vector_sets_rigid():
    # Source: Unit square in Z=0, corner at origin
    A = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=float)
    
    # Target: Translate by (10, 0, 0)
    B = A + [10, 0, 0]
    
    # Register A to B
    Tx, err = register_vector_sets(A, B, method='rigid', return_error=True)
    
    # Expected translation matrix
    # [1 0 0 10]
    # [0 1 0 0]
    # [0 0 1 0]
    # [0 0 0 1]
    
    # Transform A using Tx
    homo_A = np.c_[A, np.ones(4)]
    A_prime_homo = homo_A @ Tx.T
    A_prime = A_prime_homo[:, :3]
    
    np.testing.assert_allclose(A_prime, B, atol=1e-6)
    assert err < 1e-6

def test_get_edges_for_vertex():
    # Connectivity list (sparse like)
    # Edge 0: 0-1
    # Edge 1: 1-2
    # Edge 2: 0-2
    connections = np.array([
        [0, 1],
        [1, 2],
        [0, 2]
    ])
    
    # Vertex 0 connects to Edge 0 and Edge 2
    edges_0 = get_edges_for_vertex(connections, 0)
    assert sorted(edges_0) == [0, 2]
    
    # Vertex 1 connects to Edge 0 and Edge 1
    edges_1 = get_edges_for_vertex(connections, 1)
    assert sorted(edges_1) == [0, 1]
    
    # Vertex 3 has no edges
    edges_3 = get_edges_for_vertex(connections, 3)
    assert len(edges_3) == 0
