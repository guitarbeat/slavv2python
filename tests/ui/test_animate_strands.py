import pathlib
import sys
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
try:
    from slavv.visualization import NetworkVisualizer
except ImportError:
    from slavv.visualization import NetworkVisualizer


def build_sample_network():
    vertices = {
        'positions': np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0]]),
        'energies': np.zeros(4),
    }
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [0, 2, 0]]),
            np.array([[0, 2, 0], [1, 2, 0]]),
        ],
        'connections': np.array([[0, 1], [1, 2], [2, 3]]),
    }
    network = {
        'strands': [np.array([0, 1, 2]), np.array([2, 3])]
    }
    params = {'microns_per_voxel': [1, 1, 1]}
    return vertices, edges, network, params


def test_animate_strands_3d_frames():
    vis = NetworkVisualizer()
    vertices, edges, network, params = build_sample_network()
    fig = vis.animate_strands_3d(vertices, edges, network, params)
    assert len(fig.frames) == len(network['strands'])
    # Each frame should contain vertex scatter plus at least one edge trace
    for frame in fig.frames:
        assert len(frame.data) >= 2

