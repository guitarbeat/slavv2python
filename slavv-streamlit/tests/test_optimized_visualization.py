
import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

@pytest.fixture
def sample_data():
    num_edges = 200
    vertices = {
        'positions': np.random.rand(num_edges * 2, 3) * 100,
        'energies': np.random.rand(num_edges * 2),
        'radii_microns': np.random.rand(num_edges * 2),
    }

    edges = {
        'traces': [],
        'connections': [],
        'energies': np.random.rand(num_edges)
    }

    for i in range(num_edges):
        start = np.random.rand(3) * 100
        end = start + np.random.rand(3) * 10
        trace = np.linspace(start, end, 5)
        edges['traces'].append(trace)
        edges['connections'].append([i*2, i*2+1])

    network = {
        'strands': [[i*2, i*2+1] for i in range(num_edges)],
        'bifurcations': [0, 1], # Dummy bifurcations
        'vertex_degrees': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def test_optimized_vs_unoptimized_structure(sample_data):
    vertices, edges, network, parameters = sample_data
    viz = NetworkVisualizer()

    # Force optimized path (opacity_by=None, >100 edges)
    fig_opt = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy', opacity_by=None,
        show_bifurcations=False # Simplify for trace counting
    )

    # Vertices + 1 merged edge trace = 2 traces
    # The merged edge trace has colorbar built-in, so no extra trace needed.
    assert len(fig_opt.data) == 2
    assert isinstance(fig_opt.data[0], go.Scatter3d) # Edges
    assert isinstance(fig_opt.data[1], go.Scatter3d) # Vertices

    # Check if edges trace has None separators
    edge_trace = fig_opt.data[0]
    assert None in edge_trace.x
    assert len(edge_trace.x) > 5 * 200 # 5 points per edge * 200 edges

    # Check coloring
    assert len(edge_trace.line.color) == len(edge_trace.x)

    # Force unoptimized path by reducing edge count
    small_edges = {
        'traces': edges['traces'][:50],
        'connections': edges['connections'][:50],
        'energies': edges['energies'][:50]
    }

    fig_unopt = viz.plot_3d_network(
        vertices, small_edges, network, parameters,
        color_by='energy', opacity_by=None,
        show_bifurcations=False
    )

    # Vertices + 50 edge traces + 1 colorbar trace = 52 traces
    assert len(fig_unopt.data) == 52

def test_optimized_coloring_types(sample_data):
    vertices, edges, network, parameters = sample_data
    viz = NetworkVisualizer()

    for color_mode in ['energy', 'depth', 'radius', 'length', 'strand_id']:
        fig = viz.plot_3d_network(
            vertices, edges, network, parameters,
            color_by=color_mode, opacity_by=None,
            show_bifurcations=False
        )

        # Should be optimized
        assert len(fig.data) == 2
        edge_trace = fig.data[0]
        assert None in edge_trace.x

        # Check that color array is populated
        assert hasattr(edge_trace.line, 'color')
        assert len(edge_trace.line.color) > 0

def test_optimized_hover_text(sample_data):
    vertices, edges, network, parameters = sample_data
    viz = NetworkVisualizer()

    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy', opacity_by=None,
        show_bifurcations=False
    )

    edge_trace = fig.data[0]
    assert hasattr(edge_trace, 'text')
    assert len(edge_trace.text) == len(edge_trace.x)
    # Check content of hover text
    assert "Edge" in edge_trace.text[0]
    assert "Length" in edge_trace.text[0]
