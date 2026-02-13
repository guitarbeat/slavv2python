import pathlib
import sys
import pytest
import importlib
st = pytest.importorskip("streamlit")


def test_app_sets_wide_layout(monkeypatch):
    called = {}

    def fake_config(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(st, "set_page_config", fake_config)
    from apps import web_app as app
    importlib.reload(app)
    assert called.get("layout") == "wide"
