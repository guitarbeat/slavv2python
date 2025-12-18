
import numpy as np
import pytest
import plotly.graph_objects as go
import plotly.express as px
from src.visualization import NetworkVisualizer

def test_edge_depth_coloring():
    vis = NetworkVisualizer()
    vertices = {
        'positions': np.zeros((0, 3), dtype=float),
        'energies': np.array([], dtype=float),
        'radii': np.array([], dtype=float),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 0, 2], [0, 0, 3]], dtype=float),
        ],
        'connections': [],
        'energies': np.array([0.0, 0.0], dtype=float),
    }
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}
    network = {'bifurcations': []}

    fig = vis.plot_2d_network(
        vertices,
        edges,
        network,
        params,
        color_by='depth',
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    depths = [0.5, 2.5]
    norm = [(d - min(depths)) / (max(depths) - min(depths)) for d in depths]
    expected_colors = [px.colors.sample_colorscale('Viridis', v)[0] for v in norm]

    # Find the "Edge Data" trace
    data_trace = next(t for t in fig.data if t.name == 'Edge Data')

    # Assert colors. Note: data trace color is an array matching points.
    # Trace 0 has 2 points, Trace 1 has 2 points. Separation by NaN.
    # Points: T0_p0, T0_p1, NaN, T1_p0, T1_p1, NaN
    # Colors: C0, C0, C0, C1, C1, C1

    colors = data_trace.marker.color
    assert colors[0] == expected_colors[0]
    assert colors[1] == expected_colors[0]
    assert colors[3] == expected_colors[1]
    assert colors[4] == expected_colors[1]

def test_edge_energy_coloring():
    vis = NetworkVisualizer()
    vertices = {
        'positions': np.zeros((0, 3), dtype=float),
        'energies': np.array([], dtype=float),
        'radii': np.array([], dtype=float),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 0, 2], [0, 0, 3]], dtype=float),
        ],
        'connections': [],
        'energies': np.array([0.2, -0.1], dtype=float),
    }
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}
    network = {'bifurcations': []}

    fig = vis.plot_2d_network(
        vertices,
        edges,
        network,
        params,
        color_by='energy',
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    energies = edges['energies']
    norm = (energies - np.min(energies)) / (np.max(energies) - np.min(energies))
    expected_colors = [px.colors.sample_colorscale('RdBu_r', float(v))[0] for v in norm]

    data_trace = next(t for t in fig.data if t.name == 'Edge Data')
    colors = data_trace.marker.color

    assert colors[0] == expected_colors[0]
    assert colors[3] == expected_colors[1]

def test_edge_strand_coloring():
    vis = NetworkVisualizer()
    vertices = {
        'positions': np.zeros((5, 3), dtype=float),
        'energies': np.zeros(5, dtype=float),
        'radii': np.zeros(5, dtype=float),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 1, 0], [0, 2, 0]], dtype=float),
            np.array([[1, 0, 0], [2, 0, 0]], dtype=float),
        ],
        'connections': np.array([[0, 1], [1, 2], [3, 4]], dtype=int),
        'energies': np.zeros(3, dtype=float),
    }
    network = {'bifurcations': [], 'strands': [[0, 1, 2], [3, 4]]}
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}

    fig = vis.plot_2d_network(
        vertices,
        edges,
        network,
        params,
        color_by='strand_id',
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    data_trace = next(t for t in fig.data if t.name == 'Edge Data')
    colors = data_trace.marker.color

    # Trace 0 (points 0,1) and Trace 1 (points 3,4) belong to same strand (0)
    # Trace 2 (points 6,7) belong to strand 1

    assert colors[0] == colors[3] # Same strand
    assert colors[0] != colors[6] # Different strand

def test_edge_radius_coloring():
    vis = NetworkVisualizer()
    vertices = {
        'positions': np.zeros((3, 3), dtype=float),
        'energies': np.zeros(3, dtype=float),
        'radii_microns': np.array([2.0, 4.0, 6.0], dtype=float),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 0, 1], [0, 0, 2]], dtype=float),
        ],
        'connections': np.array([[0, 1], [1, 2]], dtype=int),
        'energies': np.zeros(2, dtype=float),
    }
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}
    network = {'bifurcations': []}

    fig = vis.plot_2d_network(
        vertices,
        edges,
        network,
        params,
        color_by='radius',
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    radii = [(2.0 + 4.0) / 2.0, (4.0 + 6.0) / 2.0]
    norm = (np.array(radii) - min(radii)) / (max(radii) - min(radii))
    expected_colors = [px.colors.sample_colorscale('Plasma', float(v))[0] for v in norm]

    data_trace = next(t for t in fig.data if t.name == 'Edge Data')
    colors = data_trace.marker.color

    assert colors[0] == expected_colors[0]
    assert colors[3] == expected_colors[1]

def test_edge_length_coloring():
    vis = NetworkVisualizer()
    vertices = {
        'positions': np.zeros((0, 3), dtype=float),
        'energies': np.array([], dtype=float),
        'radii': np.array([], dtype=float),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
        ],
        'connections': [],
        'energies': np.zeros(2, dtype=float),
    }
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}
    network = {'bifurcations': []}

    fig = vis.plot_2d_network(
        vertices,
        edges,
        network,
        params,
        color_by='length',
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    lengths = np.array([1.0, 2.0])
    norm = (lengths - lengths.min()) / (lengths.max() - lengths.min())
    expected_colors = [px.colors.sample_colorscale('Cividis', float(v))[0] for v in norm]

    data_trace = next(t for t in fig.data if t.name == 'Edge Data')
    colors = data_trace.marker.color

    assert colors[0] == expected_colors[0]
    assert colors[3] == expected_colors[1]

def test_3d_depth_opacity():
    vis = NetworkVisualizer()
    vertices = {
        'positions': np.zeros((0, 3), dtype=float),
        'energies': np.array([], dtype=float),
        'radii': np.array([], dtype=float),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 0, 2], [0, 0, 3]], dtype=float),
        ],
        'connections': [],
        'energies': np.zeros(2, dtype=float),
    }
    params = {'microns_per_voxel': [1.0, 1.0, 1.0]}
    network = {'bifurcations': []}

    fig = vis.plot_3d_network(
        vertices,
        edges,
        network,
        params,
        color_by='energy',
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
        opacity_by='depth',
    )

    # In merged 3D trace, opacity is encoded in the RGBA string.
    # Edge 0 (depth 0.5) should be more opaque (alpha ~1.0) than Edge 1 (depth 2.5).
    # dmin=0.5, dmax=2.5. norm0=0, norm1=1.
    # opacities: 1.0 - 0.8*n => op0=1.0, op1=0.2.

    # Trace name is 'Edges'.
    trace = next(t for t in fig.data if t.name == 'Edges')
    colors = trace.line.color

    # Helper to extract alpha from 'rgba(r, g, b, a)' or 'rgb...'
    def get_alpha(c_str):
        if 'rgba' in c_str:
            return float(c_str.split(',')[-1].strip(')'))
        return 1.0

    alpha0 = get_alpha(colors[0])
    alpha1 = get_alpha(colors[3])

    assert alpha0 > alpha1
    assert abs(alpha0 - 1.0) < 0.01
    assert abs(alpha1 - 0.2) < 0.01
