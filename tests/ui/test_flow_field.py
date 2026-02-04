import sys
import pathlib
import numpy as np
import plotly.graph_objects as go

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
try:
    from slavv.visualization import NetworkVisualizer
except ImportError:
    from src.slavv.visualization import NetworkVisualizer


def test_plot_flow_field_returns_cone():
    viz = NetworkVisualizer()
    edges = {"traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=float)]}
    params = {"microns_per_voxel": [1.0, 1.0, 1.0]}
    fig = viz.plot_flow_field(edges, params)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Cone) for trace in fig.data)
