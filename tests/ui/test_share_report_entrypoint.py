import pytest


def test_share_report_requires_full_network_results():
    pytest.importorskip("streamlit")
    from slavv_python.apps.shared_services.app import _has_full_network_results

    assert _has_full_network_results({"energy_data": {}}) is False
    assert _has_full_network_results({"vertices": {}, "edges": {}, "network": {}}) is True
