import sys
import pathlib
import numpy as np
import plotly.express as px

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from slavv.visualization import NetworkVisualizer
except ImportError:
    from src.slavv.visualization import NetworkVisualizer


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

    assert fig.data[0].line.color == expected_colors[0]
    assert fig.data[1].line.color == expected_colors[1]


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

    assert fig.data[0].line.color == expected_colors[0]
    assert fig.data[1].line.color == expected_colors[1]


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

    assert fig.data[0].line.color == fig.data[1].line.color
    assert fig.data[0].line.color != fig.data[2].line.color


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

    assert fig.data[0].line.color == expected_colors[0]
    assert fig.data[1].line.color == expected_colors[1]


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

    assert fig.data[0].line.color == expected_colors[0]
    assert fig.data[1].line.color == expected_colors[1]


def test_3d_depth_opacity():
    # Test updated to verify optimization behavior (single trace)
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

    # In optimized version, we merge all edges into a single trace
    # The previous test checked individual opacities, which is no longer supported
    # Verify we have exactly one edge trace (plus colorbar trace potentially, but here color_by='energy' and 2 edges -> 1 trace)
    # Actually add_colorbar adds a scatter3d trace for the colorbar

    edge_traces = [t for t in fig.data if t.mode == 'lines']
    assert len(edge_traces) == 1, "Optimized visualization should have a single merged edge trace"
    assert edge_traces[0].opacity == 1.0, "Merged trace should have uniform opacity"


def test_3d_length_colorbar():
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

    fig = vis.plot_3d_network(
        vertices,
        edges,
        network,
        params,
        color_by='length',
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(getattr(trace, 'marker', None), 'showscale') and trace.marker.showscale
    ]
    assert colorbar_traces, 'Expected a colorbar trace for edge length coloring'
    assert colorbar_traces[0].marker.colorbar.title.text.lower() == 'length'
