import importlib
import warnings

import numpy as np
import plotly.graph_objects as go
import pytest
from slavv_python.visualization import NetworkVisualizer


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


def test_app_main_runs():
    """Smoke test that Streamlit app main executes without error."""
    pytest.importorskip("streamlit")
    from slavv_python.apps.streamlit.shell import main

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()


def test_app_sets_wide_layout(monkeypatch):
    st = pytest.importorskip("streamlit")
    called = {}

    def fake_config(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(st, "set_page_config", fake_config)
    from slavv_python.apps.streamlit import app

    importlib.reload(app)
    assert called.get("layout") == "wide"
