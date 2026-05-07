import numpy as np
import plotly.express as px
import pytest
from slavv_python.visualization import NetworkVisualizer


@pytest.fixture
def base_inputs():
    return {
        "params": {"microns_per_voxel": [1.0, 1.0, 1.0]},
        "network": {"bifurcations": []},
    }


def _line_traces(fig):
    return [trace for trace in fig.data if getattr(trace, "mode", None) == "lines"]


@pytest.mark.parametrize(
    ("color_by", "colorscale", "vertices", "edges", "expected_values"),
    [
        (
            "depth",
            "Viridis",
            {"positions": np.zeros((0, 3)), "energies": np.array([]), "radii": np.array([])},
            {
                "traces": [
                    np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
                    np.array([[0, 0, 2], [0, 0, 3]], dtype=float),
                ],
                "connections": [],
                "energies": np.zeros(2, dtype=float),
            },
            np.array([0.5, 2.5]),
        ),
        (
            "energy",
            "RdBu_r",
            {"positions": np.zeros((0, 3)), "energies": np.array([]), "radii": np.array([])},
            {
                "traces": [
                    np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
                    np.array([[0, 0, 2], [0, 0, 3]], dtype=float),
                ],
                "connections": [],
                "energies": np.array([0.2, -0.1], dtype=float),
            },
            np.array([0.2, -0.1]),
        ),
        (
            "radius",
            "Plasma",
            {
                "positions": np.zeros((3, 3), dtype=float),
                "energies": np.zeros(3, dtype=float),
                "radii_microns": np.array([2.0, 4.0, 6.0], dtype=float),
            },
            {
                "traces": [
                    np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
                    np.array([[0, 0, 1], [0, 0, 2]], dtype=float),
                ],
                "connections": np.array([[0, 1], [1, 2]], dtype=int),
                "energies": np.zeros(2, dtype=float),
            },
            np.array([3.0, 5.0]),
        ),
        (
            "length",
            "Cividis",
            {"positions": np.zeros((0, 3)), "energies": np.array([]), "radii": np.array([])},
            {
                "traces": [
                    np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
                    np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
                ],
                "connections": [],
                "energies": np.zeros(2, dtype=float),
            },
            np.array([1.0, 2.0]),
        ),
    ],
)
def test_2d_numeric_edge_coloring_maps_expected_colors(
    base_inputs, color_by, colorscale, vertices, edges, expected_values
):
    vis = NetworkVisualizer()
    fig = vis.plot_2d_network(
        vertices,
        edges,
        base_inputs["network"],
        base_inputs["params"],
        color_by=color_by,
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    line_colors = [trace.line.color for trace in _line_traces(fig)]
    normalized = (expected_values - np.min(expected_values)) / (
        np.max(expected_values) - np.min(expected_values)
    )
    expected_colors = [px.colors.sample_colorscale(colorscale, float(v))[0] for v in normalized]

    assert line_colors == expected_colors


def test_2d_strand_coloring_groups_same_strand_edges():
    vis = NetworkVisualizer()
    vertices = {
        "positions": np.zeros((5, 3), dtype=float),
        "energies": np.zeros(5, dtype=float),
        "radii": np.zeros(5, dtype=float),
    }
    edges = {
        "traces": [
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 1, 0], [0, 2, 0]], dtype=float),
            np.array([[1, 0, 0], [2, 0, 0]], dtype=float),
        ],
        "connections": np.array([[0, 1], [1, 2], [3, 4]], dtype=int),
        "energies": np.zeros(3, dtype=float),
    }
    network = {"bifurcations": [], "strands": [[0, 1, 2], [3, 4]]}

    fig = vis.plot_2d_network(
        vertices,
        edges,
        network,
        {"microns_per_voxel": [1.0, 1.0, 1.0]},
        color_by="strand_id",
        projection_axis=2,
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    line_traces = _line_traces(fig)
    assert len(line_traces) == 2
    assert line_traces[0].line.color != line_traces[1].line.color


def test_3d_depth_opacity_uses_single_merged_edge_trace(base_inputs):
    vis = NetworkVisualizer()
    vertices = {
        "positions": np.zeros((0, 3), dtype=float),
        "energies": np.array([], dtype=float),
        "radii": np.array([], dtype=float),
    }
    edges = {
        "traces": [
            np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 0, 2], [0, 0, 3]], dtype=float),
        ],
        "connections": [],
        "energies": np.zeros(2, dtype=float),
    }

    fig = vis.plot_3d_network(
        vertices,
        edges,
        base_inputs["network"],
        base_inputs["params"],
        color_by="energy",
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
        opacity_by="depth",
    )

    line_traces = _line_traces(fig)
    assert len(line_traces) == 1
    assert line_traces[0].opacity == 1.0


def test_3d_length_coloring_adds_colorbar(base_inputs):
    vis = NetworkVisualizer()
    vertices = {
        "positions": np.zeros((0, 3), dtype=float),
        "energies": np.array([], dtype=float),
        "radii": np.array([], dtype=float),
    }
    edges = {
        "traces": [
            np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
        ],
        "connections": [],
        "energies": np.zeros(2, dtype=float),
    }

    fig = vis.plot_3d_network(
        vertices,
        edges,
        base_inputs["network"],
        base_inputs["params"],
        color_by="length",
        show_vertices=False,
        show_edges=True,
        show_bifurcations=False,
    )

    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(getattr(trace, "marker", None), "showscale") and trace.marker.showscale
    ]
    assert colorbar_traces
    assert colorbar_traces[0].marker.colorbar.title.text.lower() == "length"
