import sys
import pathlib
import numpy as np
import pytest
import plotly.graph_objects as go

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

from slavv.visualization import NetworkVisualizer

def test_plot_2d_network_large_optimization():
    """Test plot_2d_network with enough edges to trigger optimization path (>100)"""
    vis = NetworkVisualizer()
    n_edges = 150

    vertices = {
        'positions': np.random.rand(n_edges * 2, 3),
        'energies': np.random.rand(n_edges * 2),
        'radii': np.random.rand(n_edges * 2),
    }

    traces = []
    connections = []
    energies = []

    for i in range(n_edges):
        traces.append(np.array([[0,0,0], [1,1,1]]))
        connections.append([i, i+1])
        energies.append(0.5)

    edges = {
        'traces': traces,
        'connections': connections,
        'energies': np.array(energies),
    }

    network = {'bifurcations': [], 'strands': []}
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    fig = vis.plot_2d_network(
        vertices, edges, network, params,
        color_by='energy',
        show_edges=True
    )

    # In optimized 2D mode, traces are merged by color bin
    # We used uniform energy (0.5), so we expect few traces (likely 1 group)

    scatter_traces = [t for t in fig.data if t.mode == 'lines']
    assert len(scatter_traces) > 0
    # Check if customdata is correct (numpy array)
    assert isinstance(scatter_traces[0].customdata, np.ndarray)
    assert scatter_traces[0].customdata.shape[1] == 2 # [idx, length]

def test_plot_3d_network_optimization_validity():
    """Test plot_3d_network to ensure numpy arrays are correctly passed"""
    vis = NetworkVisualizer()
    n_edges = 10

    vertices = {
        'positions': np.random.rand(n_edges * 2, 3),
        'energies': np.random.rand(n_edges * 2),
        'radii': np.random.rand(n_edges * 2),
    }

    traces = [np.random.rand(10, 3) for _ in range(n_edges)]

    edges = {
        'traces': traces,
        'connections': [],
        'energies': np.random.rand(n_edges),
    }

    network = {'bifurcations': []}
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    fig = vis.plot_3d_network(
        vertices, edges, network, params,
        color_by='energy',
        show_edges=True
    )

    scatter_traces = [t for t in fig.data if t.mode == 'lines']
    assert len(scatter_traces) == 1

    trace = scatter_traces[0]
    # Check for NaNs
    assert np.isnan(trace.x).any(), "Should have NaNs as separators"
    assert isinstance(trace.customdata, np.ndarray)

if __name__ == "__main__":
    pytest.main([__file__])
