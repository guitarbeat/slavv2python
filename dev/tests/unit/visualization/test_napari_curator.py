from __future__ import annotations

import importlib

import numpy as np
import pytest
from slavv.visualization import napari_curator


def test_prepare_curator_payload_adds_default_status_lists():
    vertices, edges = napari_curator._prepare_curator_payload(
        {"positions": [[1, 2, 3], [4, 5, 6]]},
        {"traces": [[[1, 2, 3], [2, 3, 4]]]},
    )

    assert vertices["status"] == [True, True]
    assert edges["status"] == [True]


def test_strip_false_status_rows_filters_parallel_lists():
    curated = napari_curator._strip_false_status_rows(
        {
            "positions": [[1, 2, 3], [4, 5, 6]],
            "status": [True, False],
            "energies": [0.1, 0.9],
            "metadata": {"keep": "untouched"},
        }
    )

    assert curated["positions"] == [[1, 2, 3]]
    assert curated["status"] == [True]
    assert curated["energies"] == [0.1]
    assert curated["metadata"] == {"keep": "untouched"}


def test_build_linear_trace_preserves_endpoints():
    p1 = np.array([1.0, 1.0, 1.0])
    p2 = np.array([1.0, 4.0, 1.0])

    trace = napari_curator._build_linear_trace(p1, p2)

    assert trace[0] == [1, 1, 1]
    assert trace[-1] == [1, 4, 1]
    assert len(trace) >= 2


def test_load_napari_modules_requires_optional_dependency(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "napari":
            raise ImportError("napari missing")
        return real_import_module(name)

    monkeypatch.setattr(napari_curator.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="slavv\\[napari\\]"):
        napari_curator._load_napari_modules()
