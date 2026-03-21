import pytest

st = pytest.importorskip("streamlit")

from slavv.apps import web_app


def test_share_report_requires_full_network_results():
    assert web_app._has_full_network_results({"energy_data": {}}) is False
    assert web_app._has_full_network_results({"vertices": {}, "edges": {}, "network": {}}) is True
