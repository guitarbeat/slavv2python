import numpy as np
import plotly.graph_objects as go

from source.visualization import NetworkVisualizer


def test_create_summary_dashboard_builds_expected_panels():
    viz = NetworkVisualizer()
    processing_results = {
        "vertices": {
            "positions": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            "radii_microns": np.array([2.0, 3.0, 4.0]),
        },
        "edges": {},
        "network": {"strands": [[0, 1, 2]]},
        "parameters": {"microns_per_voxel": [1.0, 1.0, 1.0]},
    }

    fig = viz.create_summary_dashboard(processing_results)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "SLAVV Processing Summary Dashboard"
    assert len(fig.data) == 4
    assert [trace.name for trace in fig.data] == [
        "Vertices",
        "Strand Lengths",
        "Radii",
        "Vertex Count by Depth",
    ]
