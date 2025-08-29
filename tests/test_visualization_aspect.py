import pathlib
import sys
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit'))
from src.visualization import NetworkVisualizer


def test_plot_2d_network_has_equal_aspect():
    vis = NetworkVisualizer()
    vertices = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]),
        "energies": np.array([0.1, 0.2]),
        "radii": np.array([1.0, 1.0]),
    }
    edges = {
        "traces": [np.array([[0, 0, 0], [1, 1, 1]])],
        "energies": [0.5],
        "connections": np.array([[0, 1]]),
    }
    network = {"strands": [], "bifurcations": []}
    params = {"microns_per_voxel": [1, 1, 1]}

    fig = vis.plot_2d_network(vertices, edges, network, params)

    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
