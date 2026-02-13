import pathlib
import sys
import warnings
import pytest


def test_app_main_runs(tmp_path):
    """Smoke test that Streamlit app main executes without error."""
    pytest.importorskip("streamlit")
    from apps import web_app as app  # noqa: F401
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app.main()

