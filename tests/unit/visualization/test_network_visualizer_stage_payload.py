"""Tests for NetworkVisualizer Stage Result normalization and theme policy."""

from __future__ import annotations

import numpy as np

from slavv_python.schema.results import VertexSet
from slavv_python.visualization.network_plots import (
    NETWORK_COLOR_SCHEMES,
    NetworkVisualizer,
    stage_payload,
)


def test_stage_payload_accepts_vertex_set() -> None:
    vertices = VertexSet.create(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        np.array([0, 1], dtype=np.int32),
        np.array([-1.0, -2.0], dtype=np.float64),
        np.array([1.0, 1.5], dtype=np.float64),
        np.array([1.0, 1.5], dtype=np.float64),
    )
    payload = stage_payload(vertices)
    assert "positions" in payload
    assert len(payload["positions"]) == 2


def test_stage_payload_accepts_mapping_and_none() -> None:
    assert stage_payload(None) == {}
    assert stage_payload({"a": 1}) == {"a": 1}


def test_visualizer_theme_defaults_and_resolve_color_by() -> None:
    viz = NetworkVisualizer(default_color_by="depth")
    assert viz.color_schemes["energy"] == NETWORK_COLOR_SCHEMES["energy"]
    assert viz.resolve_color_by(None) == "depth"
    assert viz.resolve_color_by("radius") == "radius"
    assert viz.resolve_color_by("not-a-scheme") == "depth"
