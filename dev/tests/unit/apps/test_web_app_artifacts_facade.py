import importlib
import sys
import types
import typing


def _install_fake_streamlit(monkeypatch):
    fake_streamlit = types.SimpleNamespace(
        cache_data=lambda *args, **kwargs: lambda func: func,
        dialog=lambda *args, **kwargs: lambda func: func,
        fragment=lambda *args, **kwargs: lambda func: func,
        set_page_config=lambda *args, **kwargs: None,
        html=lambda *args, **kwargs: None,
        session_state={},
        download_button=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    if not hasattr(typing, "TypedDict"):

        class TypedDict(dict):
            pass

        monkeypatch.setattr(typing, "TypedDict", TypedDict, raising=False)


def test_web_app_artifact_helpers_are_imported_from_sibling_module(monkeypatch):
    _install_fake_streamlit(monkeypatch)
    sys.modules.pop("source.apps.app_services", None)
    sys.modules.pop("source.apps.web_app", None)
    sys.modules.pop("source.apps.web_app_artifacts", None)

    web_app = importlib.import_module("source.apps.web_app")
    app_services = importlib.import_module("source.apps.app_services")
    artifacts = importlib.import_module("source.apps.web_app_artifacts")

    assert web_app.cached_load_tiff_volume is app_services.cached_load_tiff_volume
    assert web_app.generate_export_data is artifacts.generate_export_data
    assert web_app.generate_share_report_data is artifacts.generate_share_report_data
    assert web_app._has_full_network_results is artifacts._has_full_network_results
    assert web_app._log_share_report_prepared_once is artifacts._log_share_report_prepared_once
    assert web_app._build_processing_run_dir is artifacts._build_processing_run_dir
