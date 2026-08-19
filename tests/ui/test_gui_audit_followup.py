"""Follow-up coverage for Streamlit GUI audit items G5-G18."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import pytest

pytest.importorskip("streamlit")

from slavv_python.interface.streamlit.empty_state import (
    MSG_NEED_EDGES,
    MSG_NEED_NETWORK,
    MSG_NO_RUN,
)
from slavv_python.interface.streamlit.shell import PAGE_HANDLERS
from slavv_python.interface.streamlit.views.curation import desktop_curator_available
from slavv_python.interface.streamlit.views.processing import available_public_energy_methods
from slavv_python.interface.streamlit.views.visualization import _apply_figure_display


def test_nav_labels_are_plain_language() -> None:
    assert list(PAGE_HANDLERS) == [
        "Home",
        "Image Processing",
        "Curation",
        "Visualization",
        "Analysis",
        "About",
    ]


def test_empty_state_copy_points_at_image_processing() -> None:
    assert "Image Processing" in MSG_NO_RUN
    assert "Image Processing" in MSG_NEED_EDGES
    assert "Image Processing" in MSG_NEED_NETWORK


def test_cupy_energy_method_hidden_when_cupy_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "slavv_python.interface.streamlit.views.processing.util.find_spec",
        lambda name: None,
    )
    methods = available_public_energy_methods()
    assert "hessian" in methods
    assert "cupy_hessian" not in methods
    assert "simpleitk_objectness" not in methods


def test_desktop_curator_respects_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVV_DISABLE_DESKTOP_CURATOR", "1")
    assert desktop_curator_available() is False
    monkeypatch.delenv("SLAVV_DISABLE_DESKTOP_CURATOR", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    if os.name != "nt":
        monkeypatch.setattr("slavv_python.interface.streamlit.views.curation.sys.platform", "linux")
        assert desktop_curator_available() is False


def test_apply_figure_display_sets_opacity_and_camera() -> None:
    fig = go.Figure(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="markers"))
    _apply_figure_display(fig, opacity=0.4, camera="Top")
    assert fig.data[0].opacity == 0.4
    assert fig.layout.scene.camera.eye.z == pytest.approx(2.4)
