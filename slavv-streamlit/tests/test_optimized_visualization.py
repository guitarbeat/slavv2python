
import numpy as np
import pytest
import sys
import os

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization import NetworkVisualizer

@pytest.fixture
def sample_data():
    """Create sample network data for testing"""
    # 3 edges, 2 vertices
    edges = {
        'traces': [
            np.array([[0,0,0], [1,1,1]]),
            np.array([[1,1,1], [2,2,2]]),
            np.array([[2,2,2], [3,3,3], [4,4,4]]) # Longer edge
        ],
        'connections': [(0,1), (1,2), (2,3)],
        'energies': [0.1, 0.5, 0.9]
    }
    vertices = {
        'positions': np.array([[0,0,0], [1,1,1], [2,2,2], [4,4,4]]),
        'energies': np.array([0.1, 0.5, 0.9, 0.2]),
        'radii_microns': np.array([1.0, 1.0, 1.0, 1.0])
    }
    network = {
        'strands': [[0, 1, 2, 3]], # One strand
        'bifurcations': [1]
    }
    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }
    return vertices, edges, network, parameters

def test_plot_3d_network_merged_structure(sample_data):
    """Test that plot_3d_network creates merged traces"""
    vertices, edges, network, parameters = sample_data
    viz = NetworkVisualizer()

    # Run plot
    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=False,
        show_bifurcations=False # Disable bifurcations to ensure only edge trace
    )

    # Check number of traces
    # Should be 1 (merged edges)
    assert len(fig.data) == 1

    trace = fig.data[0]
    assert trace.type == 'scatter3d'
    assert trace.mode == 'lines'

    # Check that data contains Nones
    assert None in trace.x
    assert None in trace.y
    assert None in trace.z

    # Check coordinate length
    # trace 0: 2 points -> 2 + 1 None = 3
    # trace 1: 2 points -> 2 + 1 None = 3
    # trace 2: 3 points -> 3 + 1 None = 4
    # Total = 10
    assert len(trace.x) == 10
    assert len(trace.line.color) == 10

def test_plot_3d_network_color_mapping(sample_data):
    """Test that color mapping works in merged trace"""
    vertices, edges, network, parameters = sample_data
    viz = NetworkVisualizer()

    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=False,
        show_bifurcations=False
    )

    trace = fig.data[0]
    colors = trace.line.color

    # Check that colors correspond to edge energies
    # trace 0 energy 0.1
    # trace 1 energy 0.5
    # trace 2 energy 0.9

    # Points 0,1 should have 0.1
    assert colors[0] == 0.1
    assert colors[1] == 0.1

    # Points 3,4 should have 0.5
    assert colors[3] == 0.5
    assert colors[4] == 0.5

    # Points 6,7,8 should have 0.9
    assert colors[6] == 0.9
    assert colors[8] == 0.9

def test_plot_3d_network_strand_coloring(sample_data):
    """Test strand_id coloring"""
    vertices, edges, network, parameters = sample_data
    viz = NetworkVisualizer()

    fig = viz.plot_3d_network(
        vertices, edges, network, parameters,
        color_by='strand_id',
        show_vertices=False,
        show_bifurcations=False
    )

    trace = fig.data[0]
    colors = trace.line.color

    # All edges belong to strand 0
    unique_colors = set(colors)
    assert len(unique_colors) == 1
    assert 0 in unique_colors

    # Check colorscale is present (Plotly expands named scales to tuples)
    assert trace.line.colorscale is not None
    # Verify it is likely Turbo (just check it is a sequence)
    assert len(trace.line.colorscale) > 0
