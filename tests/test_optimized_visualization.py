
import sys
import pathlib
import numpy as np
import pytest
import plotly.graph_objects as go

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit'))

from src.visualization import NetworkVisualizer

def test_optimized_2d_plot():
    """Test that plot_2d_network uses optimized rendering for large datasets."""
    vis = NetworkVisualizer()
    n_edges = 150

    # Create synthetic data
    vertices = {
        'positions': np.random.rand(10, 3) * 100,
        'energies': np.random.rand(10),
        'radii': np.random.rand(10),
    }

    # All edges have same energy to produce 1 group/trace
    traces = []
    for _ in range(n_edges):
        traces.append(np.array([[0,0,0], [1,1,1]]))

    edges = {
        'traces': traces,
        'connections': [],
        'energies': np.ones(n_edges) * 0.5, # Same energy -> Same color
    }

    network = {'bifurcations': []}
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    fig = vis.plot_2d_network(
        vertices, edges, network, params,
        color_by='energy',
        show_vertices=False,
        show_edges=True
    )

    # Expectation:
    # Unoptimized: 150 traces + colorbar (if valid)
    # Optimized: 1 trace (since all same color) + colorbar

    edge_traces = [t for t in fig.data if isinstance(t, (go.Scatter, go.Scattergl)) and t.mode == 'lines']

    # Check if optimized
    # Note: Optimization uses go.Scattergl
    gl_traces = [t for t in edge_traces if isinstance(t, go.Scattergl)]

    assert len(edge_traces) < n_edges, f"Expected fewer than {n_edges} traces, got {len(edge_traces)}"
    assert len(gl_traces) > 0, "Expected go.Scattergl traces for optimized plot"
    # Should be 1 trace since color is uniform
    assert len(gl_traces) == 1, f"Expected 1 merged trace for uniform color, got {len(gl_traces)}"

def test_optimized_3d_plot():
    """Test that plot_3d_network uses optimized rendering for large datasets."""
    vis = NetworkVisualizer()
    n_edges = 150

    # Create synthetic data
    vertices = {
        'positions': np.random.rand(10, 3) * 100,
        'energies': np.random.rand(10),
        'radii': np.random.rand(10),
    }

    traces = []
    for _ in range(n_edges):
        traces.append(np.array([[0,0,0], [1,1,1]]))

    edges = {
        'traces': traces,
        'connections': [],
        'energies': np.random.rand(n_edges), # Random energy -> Varied colors
    }

    network = {'bifurcations': []}
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    fig = vis.plot_3d_network(
        vertices, edges, network, params,
        color_by='energy',
        show_vertices=False,
        show_edges=True
    )

    # Expectation:
    # Unoptimized: 150 traces
    # Optimized: 1 merged trace (regardless of colors, as we use array color)

    edge_traces = [t for t in fig.data if isinstance(t, go.Scatter3d) and t.mode == 'lines']

    assert len(edge_traces) == 1, f"Expected 1 merged trace, got {len(edge_traces)}"

    # Verify color array size matches data size
    trace = edge_traces[0]
    # Each edge has 2 points + 1 None = 3 points. Total = 150 * 3 = 450 points.
    expected_points = n_edges * 3
    assert len(trace.x) == expected_points
    assert len(trace.line.color) == expected_points
