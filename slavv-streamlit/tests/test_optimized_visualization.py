
import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization import NetworkVisualizer

def test_plot_3d_network_optimization():
    """Test that 3D network plotting uses optimization for large datasets."""
    visualizer = NetworkVisualizer()

    # Create a large synthetic dataset (> 100 edges)
    n_edges = 150
    points_per_edge = 5

    vertices = {
        'positions': np.random.rand(n_edges * 2, 3),
        'energies': np.random.rand(n_edges * 2),
        'radii': np.random.rand(n_edges * 2)
    }

    edges = {
        'traces': [np.random.rand(points_per_edge, 3) for _ in range(n_edges)],
        'energies': np.random.rand(n_edges),
        'connections': [(i, i+1) for i in range(0, n_edges * 2, 2)]
    }

    network = {
        'bifurcations': [],
        'strands': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    # 1. Test Optimized Path (default conditions: > 100 edges, opacity_by=None)
    fig_opt = visualizer.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=False, # Simplify check
        show_edges=True,
        show_bifurcations=False
    )

    # Expectation:
    # - 1 trace for edges (merged)
    # - 0 trace for vertices (disabled)
    # - 0 trace for bifurcations (disabled)
    # - 0 trace for colorbar (merged trace handles it)
    # Total traces should be 1.

    assert len(fig_opt.data) == 1, f"Expected 1 trace (optimized), got {len(fig_opt.data)}"
    assert isinstance(fig_opt.data[0], go.Scatter3d)
    assert fig_opt.data[0].mode == 'lines'

    # Check that coordinate arrays contain None (separators)
    # n_edges lines means n_edges separators? Or n_edges - 1?
    # My implementation appends None after EVERY edge.
    assert fig_opt.data[0].x.count(None) == n_edges

    # 2. Test Unoptimized Path (few edges)
    n_small = 10
    edges_small = {
        'traces': [np.random.rand(points_per_edge, 3) for _ in range(n_small)],
        'energies': np.random.rand(n_small),
        'connections': [(i, i+1) for i in range(0, n_small * 2, 2)]
    }
    vertices_small = {
        'positions': np.random.rand(n_small * 2, 3),
        'energies': np.random.rand(n_small * 2),
        'radii': np.random.rand(n_small * 2)
    }

    fig_unopt = visualizer.plot_3d_network(
        vertices_small, edges_small, network, parameters,
        color_by='energy',
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False
    )

    # Expectation:
    # - n_small traces for edges
    # - 1 trace for colorbar (unoptimized path adds separate colorbar trace)
    # Total traces should be n_small + 1
    assert len(fig_unopt.data) == n_small + 1, f"Expected {n_small + 1} traces (unoptimized), got {len(fig_unopt.data)}"

    # 3. Test Unoptimized Path (opacity_by='depth')
    # Even with many edges, it should not optimize if opacity varies
    fig_opacity = visualizer.plot_3d_network(
        vertices, edges, network, parameters, # Uses large dataset
        color_by='energy',
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
        opacity_by='depth'
    )

    # Expectation:
    # - n_edges traces for edges
    # - 1 trace for colorbar
    # Total traces should be n_edges + 1
    assert len(fig_opacity.data) == n_edges + 1, f"Expected {n_edges + 1} traces (opacity unoptimized), got {len(fig_opacity.data)}"
