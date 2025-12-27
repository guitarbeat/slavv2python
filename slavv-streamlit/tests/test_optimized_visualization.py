
import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

def create_mock_data(n_edges=1000):
    vertices = {
        'positions': np.random.rand(n_edges * 2, 3),
        'energies': np.random.rand(n_edges * 2),
        'radii': np.random.rand(n_edges * 2),
        'radii_microns': np.random.rand(n_edges * 2)
    }

    edges = {
        'traces': [],
        'connections': [],
        'energies': np.random.rand(n_edges)
    }

    for i in range(n_edges):
        start = np.random.rand(3)
        end = np.random.rand(3)
        path = np.linspace(start, end, 5)
        edges['traces'].append(path)
        edges['connections'].append([i*2, i*2+1])

    network = {
        'strands': [],
        'bifurcations': [],
        'vertex_degrees': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def test_optimized_3d_plot_trace_count():
    viz = NetworkVisualizer()
    n_edges = 200
    vertices, edges, network, parameters = create_mock_data(n_edges=n_edges)

    fig = viz.plot_3d_network(vertices, edges, network, parameters)

    # Check trace count. Should be small (e.g. vertices + bifurcations + 1 for edges)
    # Standard implementation would be n_edges + ...
    trace_count = len(fig.data)

    # We expect significantly fewer traces than edges
    assert trace_count < n_edges, f"Expected < {n_edges} traces, got {trace_count}"

    # Specifically, we expect: 1 merged edge trace + 1 vertex trace + optional bifurcation trace + optional colorbar traces
    # It should be around 3-5
    assert trace_count <= 5, f"Expected optimized trace count <= 5, got {trace_count}"

def test_standard_3d_plot_trace_count():
    viz = NetworkVisualizer()
    n_edges = 50 # Below threshold
    vertices, edges, network, parameters = create_mock_data(n_edges=n_edges)

    fig = viz.plot_3d_network(vertices, edges, network, parameters)

    # Standard implementation creates trace per edge
    trace_count = len(fig.data)

    # We expect roughly n_edges traces (+ extras)
    assert trace_count >= n_edges, f"Expected >= {n_edges} traces, got {trace_count}"

def test_optimized_3d_plot_properties():
    viz = NetworkVisualizer()
    n_edges = 200
    vertices, edges, network, parameters = create_mock_data(n_edges=n_edges)

    fig = viz.plot_3d_network(vertices, edges, network, parameters, color_by='energy')

    # Find the edge trace (it's a scatter3d with mode='lines')
    edge_traces = [t for t in fig.data if isinstance(t, go.Scatter3d) and t.mode == 'lines']
    assert len(edge_traces) == 1, "Should have exactly one merged edge trace"

    trace = edge_traces[0]

    # Check if x, y, z contains NaNs (separators)
    assert np.isnan(trace.x).any(), "Merged trace should contain NaNs as separators"

    # Check coloring
    # line.color should be an array/list
    assert isinstance(trace.line.color, (list, np.ndarray, tuple)), "Line color should be an array"
    assert len(trace.line.color) == len(trace.x), "Color array length should match coordinate length"

if __name__ == "__main__":
    test_optimized_3d_plot_trace_count()
    test_standard_3d_plot_trace_count()
    test_optimized_3d_plot_properties()
