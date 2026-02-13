import pathlib
import sys
import pytest
st = pytest.importorskip("streamlit")


def test_app_sets_wide_layout(monkeypatch):
    called = {}

    def fake_config(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(st, "set_page_config", fake_config)
    if 'app' in sys.modules:
        del sys.modules['app']
    import app  # noqa: F401
    assert called.get("layout") == "wide"
