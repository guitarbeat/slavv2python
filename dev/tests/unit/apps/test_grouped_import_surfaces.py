from __future__ import annotations

from source.apps.cli import main as cli_main
from source.apps.streamlit import main as streamlit_main
from source.apps.streamlit.pages.processing import show_processing_page
from source.apps.streamlit.services.exports import generate_export_data
from source.apps.streamlit.state.analysis import normalize_analysis_results


def test_grouped_app_import_surfaces_export_expected_symbols():
    assert callable(cli_main)
    assert callable(streamlit_main)
    assert callable(show_processing_page)
    assert callable(generate_export_data)
    assert callable(normalize_analysis_results)
