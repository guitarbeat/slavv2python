import pytest


def test_share_report_requires_full_network_results():
    pytest.importorskip("streamlit")
    from slavv.apps import web_app

    assert web_app._has_full_network_results({"energy_data": {}}) is False
    assert web_app._has_full_network_results({"vertices": {}, "edges": {}, "network": {}}) is True
