
import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

@pytest.fixture
def mock_network_data():
    # Create a small network
    vertices = {
        'positions': np.array([[0,0,0], [10,0,0], [20,0,0], [10,10,0]]),
        'energies': np.array([0.1, 0.2, 0.3, 0.4]),
        'radii': np.array([1.0, 1.0, 1.0, 1.0]),
        'radii_microns': np.array([1.0, 1.0, 1.0, 1.0])
    }

    edges = {
        'traces': [
            np.array([[0,0,0], [5,0,0], [10,0,0]]),
            np.array([[10,0,0], [15,0,0], [20,0,0]]),
            np.array([[10,0,0], [10,5,0], [10,10,0]])
        ],
        'connections': [(0,1), (1,2), (1,3)],
        'energies': [0.15, 0.25, 0.3]
    }

    network = {
        'strands': [[0, 1, 2], [1, 3]], # Dummy strands
        'bifurcations': [1]
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def test_plot_3d_network_optimized_trace_count(mock_network_data):
    """
    Verify that plot_3d_network uses optimized rendering (merged traces)
    when opacity_by is None.
    """
    vertices, edges, network, parameters = mock_network_data
    viz = NetworkVisualizer()

    # Render with optimization (default opacity_by=None)
    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=True,
        show_edges=True,
        show_bifurcations=True
    )

    # Expected traces:
    # 1. Merged Edges
    # 2. Colorbar (Scatter3d dummy)
    # 3. Vertices
    # 4. Bifurcations
    # Total = 4

    # Note: plot_3d_network adds vertices and bifurcations as separate traces.
    # The optimization merges only edges.

    # Check trace count.
    # Without optimization, we would have 3 edges + 1 vertices + 1 bifurcations + 1 colorbar = 6 traces.
    # With optimization, we should have 1 merged edge trace + 1 vertices + 1 bifurcations + 1 colorbar = 4 traces.

    assert len(fig.data) == 4

    # Check that one trace is the merged edge trace
    edge_traces = [t for t in fig.data if t.name == 'Network']
    assert len(edge_traces) == 1
    assert len(edge_traces[0].x) > 10 # Should contain multiple points + None separators

def test_plot_3d_network_fallback(mock_network_data):
    """
    Verify that plot_3d_network falls back to individual traces
    when opacity_by is set.
    """
    vertices, edges, network, parameters = mock_network_data
    viz = NetworkVisualizer()

    # Render without optimization (opacity_by='depth')
    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        opacity_by='depth',
        show_vertices=False, # Simplify check
        show_bifurcations=False
    )

    # Expected traces: 3 edges + 1 colorbar = 4 traces
    # With optimization it would be 1 + 1 = 2.

    assert len(fig.data) == 4

    edge_trace_names = [t.name for t in fig.data]
    assert 'Edge 0' in edge_trace_names
    assert 'Edge 1' in edge_trace_names
    assert 'Edge 2' in edge_trace_names

def test_plot_3d_network_strand_coloring(mock_network_data):
    """
    Verify optimization works with strand_id coloring
    """
    vertices, edges, network, parameters = mock_network_data
    viz = NetworkVisualizer()

    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='strand_id',
        show_vertices=False,
        show_bifurcations=False
    )

    # Expected: 1 merged trace (no colorbar for strand_id usually? Wait, code doesn't add colorbar for strand_id if using discrete check?)
    # Let's check code:
    # if color_by in {'depth', 'energy', 'radius', 'length'} ... add_colorbar
    # strand_id is not in that set.

    # So expected traces: 1 (merged edges)

    assert len(fig.data) == 1
    assert fig.data[0].name == 'Network'
    # Check if colorscale is set (we used 'Turbo' or something)
    assert fig.data[0].line.colorscale is not None
