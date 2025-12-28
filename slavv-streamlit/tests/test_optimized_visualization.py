
import numpy as np
import pytest
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

def test_plot_3d_network_optimization():
    visualizer = NetworkVisualizer()

    # Generate synthetic data with enough edges to trigger optimization
    n_edges = 150
    points_per_edge = 5

    vertices = {
        'positions': np.random.rand(n_edges * 2, 3) * 100,
        'energies': np.random.rand(n_edges * 2),
        'radii': np.random.rand(n_edges * 2)
    }

    edges = {
        'traces': [np.random.rand(points_per_edge, 3) * 100 for _ in range(n_edges)],
        'energies': np.random.rand(n_edges),
        'connections': [[i, i+1] for i in range(n_edges)]
    }

    network = {
        'bifurcations': [],
        'strands': [
            [i, i+1] for i in range(n_edges)
        ]
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    # Case 1: Optimized (opacity_by=None, n_edges > 100)
    fig_opt = visualizer.plot_3d_network(vertices, edges, network, parameters, color_by='energy')

    # Check that we have fewer traces than edges
    # Traces should be: Edges (1) + Vertices (1) + Colorbar/Dummy (1) = ~3
    assert len(fig_opt.data) < 50, f"Expected < 50 traces, got {len(fig_opt.data)}"

    # Verify the edge trace properties
    edge_trace = None
    for trace in fig_opt.data:
        if isinstance(trace, go.Scatter3d) and trace.mode == 'lines':
            edge_trace = trace
            break

    assert edge_trace is not None
    assert len(edge_trace.x) > n_edges * points_per_edge # Includes None separators
    assert len(edge_trace.line.color) == len(edge_trace.x)
    assert 'customdata' in edge_trace

    # Case 2: Unoptimized (opacity_by='depth')
    fig_unopt = visualizer.plot_3d_network(vertices, edges, network, parameters, color_by='depth', opacity_by='depth')
    assert len(fig_unopt.data) >= n_edges

    # Case 3: Unoptimized (few edges)
    edges_small = {
        'traces': edges['traces'][:50],
        'energies': edges['energies'][:50],
        'connections': edges['connections'][:50]
    }
    fig_small = visualizer.plot_3d_network(vertices, edges_small, network, parameters, color_by='energy')
    assert len(fig_small.data) >= 50

    # Case 4: Optimized with strand_id (regression check)
    fig_strand = visualizer.plot_3d_network(vertices, edges, network, parameters, color_by='strand_id')
    assert len(fig_strand.data) < 50
    # Find edge trace
    strand_trace = [t for t in fig_strand.data if isinstance(t, go.Scatter3d) and t.mode == 'lines'][0]
    assert strand_trace.line.colorscale is not None
    assert len(strand_trace.line.color) == len(strand_trace.x)

if __name__ == "__main__":
    test_plot_3d_network_optimization()
