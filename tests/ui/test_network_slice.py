import numpy as np


from slavv.visualization import NetworkVisualizer


def test_plot_network_slice_filters_edges_and_vertices():
    vis = NetworkVisualizer()
    vertices = {
        "positions": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 5]]),
        "energies": np.array([0.1, 0.2, 0.3]),
        "radii": np.array([1.0, 1.0, 1.0]),
    }
    edges = {
        "traces": [
            np.array([[0, 0, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [0, 0, 5]]),
        ],
        "energies": [0.5, 0.6],
        "connections": np.array([[0, 1], [1, 2]]),
    }
    network = {"strands": []}
    params = {"microns_per_voxel": [1, 1, 1]}

    fig = vis.plot_network_slice(
        vertices, edges, network, params, axis=2, center_in_microns=0.5, thickness_in_microns=1.0
    )

    line_traces = [t for t in fig.data if t.mode == "lines"]
    marker_traces = [t for t in fig.data if t.mode == "markers"]
    assert len(line_traces) == 1
    assert len(marker_traces) == 1
    # Two vertices fall within the slice
    assert len(marker_traces[0].x) == 2
    # Axes should share scale to preserve physical aspect ratio
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1

