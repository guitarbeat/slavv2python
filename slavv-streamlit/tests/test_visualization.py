
import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

@pytest.fixture
def mock_data():
    n_vertices = 100
    n_edges = 200 # Enough to trigger optimization (> 100)

    vertices = {
        'positions': np.random.rand(n_vertices, 3) * 100,
        'energies': np.random.rand(n_vertices),
        'radii': np.random.rand(n_vertices),
        'radii_microns': np.random.rand(n_vertices),
        'scales': np.ones(n_vertices)
    }

    edges = {
        'traces': [],
        'connections': [],
        'energies': np.random.rand(n_edges)
    }

    for i in range(n_edges):
        n_points = 5
        trace = np.random.rand(n_points, 3) * 100
        edges['traces'].append(trace)
        edges['connections'].append((0, 1))

    network = {
        'strands': [[0, 1]] * n_edges,
        'bifurcations': [],
        'vertex_degrees': np.zeros(n_vertices)
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def test_plot_2d_network_optimization(mock_data):
    vertices, edges, network, parameters = mock_data
    visualizer = NetworkVisualizer()

    # Test with energy (continuous)
    fig = visualizer.plot_2d_network(
        vertices, edges, network, parameters,
        color_by='energy', show_vertices=False, show_edges=True
    )

    # Expect much fewer traces than edges due to binning
    # 64 bins + maybe colorbar/etc.
    assert len(fig.data) < len(edges['traces'])
    assert len(fig.data) <= 65 # 64 bins + 1 colorbar trace potentially

    # Check trace type
    assert isinstance(fig.data[0], go.Scattergl)

def test_plot_3d_network_optimization(mock_data):
    vertices, edges, network, parameters = mock_data
    visualizer = NetworkVisualizer()

    # Test with energy (continuous)
    fig = visualizer.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy', show_vertices=False, show_edges=True
    )

    # Expect fewer traces
    assert len(fig.data) < len(edges['traces'])
    assert len(fig.data) <= 65

    # Check trace type
    assert isinstance(fig.data[0], go.Scatter3d)

def test_small_network_no_optimization():
    # Test that small networks use standard rendering
    visualizer = NetworkVisualizer()
    n_edges = 10
    edges = {
        'traces': [np.random.rand(5, 3) for _ in range(n_edges)],
        'connections': [(0, 1)] * n_edges,
        'energies': np.random.rand(n_edges)
    }
    vertices = {
        'positions': np.random.rand(20, 3),
        'energies': np.random.rand(20)
    }
    network = {'bifurcations': []}
    parameters = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    fig = visualizer.plot_2d_network(
        vertices, edges, network, parameters,
        color_by='energy', show_vertices=False, show_edges=True
    )

    # Should have one trace per edge (plus colorbar maybe)
    # The code adds traces one by one.
    # Note: Colorbar adds 1 trace.
    assert len(fig.data) >= n_edges
    assert isinstance(fig.data[0], go.Scatter) # Not Scattergl
