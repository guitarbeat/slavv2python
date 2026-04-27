import warnings

import pytest


def test_app_main_runs():
    """Smoke test that Streamlit app main executes without error."""
    pytest.importorskip("streamlit")
    from source.apps import web_app as app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app.main()
