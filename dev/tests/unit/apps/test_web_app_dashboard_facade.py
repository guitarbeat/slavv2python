import importlib
import sys
import types
import typing

import pandas as pd


def _install_fake_streamlit(monkeypatch):
    fake_streamlit = types.SimpleNamespace(
        cache_data=lambda *args, **kwargs: (lambda func: func),
        dialog=lambda *args, **kwargs: (lambda func: func),
        fragment=lambda *args, **kwargs: (lambda func: func),
        set_page_config=lambda *args, **kwargs: None,
        html=lambda *args, **kwargs: None,
        session_state={},
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    if not hasattr(typing, "TypedDict"):
        class TypedDict(dict):
            pass

        monkeypatch.setattr(typing, "TypedDict", TypedDict, raising=False)


def test_web_app_dashboard_helpers_are_imported_from_sibling_module(monkeypatch):
    _install_fake_streamlit(monkeypatch)
    sys.modules.pop("slavv.apps.web_app", None)
    sys.modules.pop("slavv.apps.web_app_dashboard", None)

    web_app = importlib.import_module("slavv.apps.web_app")
    dashboard = importlib.import_module("slavv.apps.web_app_dashboard")

    assert web_app.DASHBOARD_PLACEHOLDER == dashboard.DASHBOARD_PLACEHOLDER
    assert web_app.DASHBOARD_BREAKDOWN_SECTIONS == dashboard.DASHBOARD_BREAKDOWN_SECTIONS
    assert web_app._dashboard_stage_frame is dashboard._dashboard_stage_frame
    assert web_app._dashboard_breakdown_frame is dashboard._dashboard_breakdown_frame
    frame = dashboard._dashboard_stage_frame(None)
    assert list(frame["Stage"]) == ["Energy", "Vertices", "Edges", "Network"]


def test_dashboard_pure_helpers_cover_filtering_and_backlog_cases():
    dashboard = importlib.import_module("slavv.apps.web_app_dashboard")

    frame = pd.DataFrame(
        [
            {"Section": "Pipeline", "Status": "live", "Metric": "Energy"},
            {"Section": "Optional Tasks", "Status": "placeholder", "Metric": "Tracked tasks"},
            {"Section": "Network", "Status": "live", "Metric": "Strands"},
            {"Section": "Share Report", "Status": "idle", "Metric": "Requested"},
        ]
    )

    filtered = dashboard.filter_dashboard_breakdown(
        frame,
        focus="Pipeline",
        selected_sections=["Pipeline", "Optional Tasks"],
        show_placeholders=False,
    )
    assert filtered["Section"].tolist() == ["Pipeline"]

    assert dashboard.normalize_dashboard_sections(None) == list(
        dashboard.DASHBOARD_BREAKDOWN_SECTIONS
    )
    assert dashboard.normalize_dashboard_sections("Pipeline") == ["Pipeline"]

    backlog = dashboard.build_dashboard_backlog_frame(
        [{"metric": "Export success rate", "owner": "Operations", "priority": "Low", "notes": ""}],
        repo_url="https://repo.example",
        release_url="https://release.example",
    )
    assert "Export success rate" in backlog["Metric"].tolist()
