
import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

@pytest.fixture
def mock_network_data():
    n_edges = 200
    n_vertices = n_edges + 1
    positions = np.random.rand(n_vertices, 3) * 100
    energies = np.random.rand(n_vertices)

    vertices = {
        'positions': positions,
        'energies': energies,
        'radii': np.random.rand(n_vertices),
        'radii_microns': np.random.rand(n_vertices)
    }

    traces = []
    connections = []

    for i in range(n_edges):
        start = positions[i]
        end = positions[i+1]
        mid = (start + end) / 2
        trace = np.vstack([start, mid, end])
        traces.append(trace)
        connections.append((i, i+1))

    edges = {
        'traces': traces,
        'connections': connections,
        'energies': np.random.rand(n_edges)
    }

    network = {
        'bifurcations': [],
        'strands': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def test_optimized_3d_network_plot(mock_network_data):
    vertices, edges, network, parameters = mock_network_data
    viz = NetworkVisualizer()

    # Test optimized path (default opacity_by=None)
    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False
    )

    # With > 100 edges and default opacity, we expect optimization
    # The figure should have fewer traces than edges
    # We expect 1 trace for network lines (merged) + maybe colorbar trace

    # Filter traces to find Scatter3d lines
    line_traces = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d) and t.mode == 'lines'
    ]

    # In optimized mode, all edges are merged into one trace
    assert len(line_traces) == 1
    assert len(line_traces[0].x) > len(edges['traces']) * 3 # Should have all points + separators

def test_unoptimized_fallback(mock_network_data):
    vertices, edges, network, parameters = mock_network_data
    viz = NetworkVisualizer()

    # Test fallback path (opacity_by='depth')
    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
        opacity_by='depth'
    )

    line_traces = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d) and t.mode == 'lines'
    ]

    # In unoptimized mode, we have one trace per edge
    assert len(line_traces) == len(edges['traces'])

def test_small_network_fallback():
    # Create small network (< 100 edges)
    n_edges = 10
    n_vertices = n_edges + 1
    positions = np.random.rand(n_vertices, 3) * 100
    vertices = {
        'positions': positions,
        'energies': np.random.rand(n_vertices)
    }
    traces = [np.vstack([positions[i], positions[i+1]]) for i in range(n_edges)]
    edges = {
        'traces': traces,
        'connections': [(i, i+1) for i in range(n_edges)],
        'energies': np.random.rand(n_edges)
    }
    network = {'bifurcations': [], 'strands': []}
    parameters = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    viz = NetworkVisualizer()
    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        show_vertices=False
    )

    line_traces = [
        t for t in fig.data
        if isinstance(t, go.Scatter3d) and t.mode == 'lines'
    ]

    # Should fallback to individual traces for small networks
    assert len(line_traces) == n_edges
