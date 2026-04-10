import pytest

pytest.importorskip("streamlit")

from slavv.apps import web_app
from slavv.runtime.run_state import RunSnapshot, StageSnapshot, TaskSnapshot


def test_dashboard_stage_frame_uses_placeholders_without_snapshot():
    frame = web_app._dashboard_stage_frame(None)

    assert frame["Stage"].tolist() == ["Energy", "Vertices", "Edges", "Network"]
    assert frame["Progress (%)"].tolist() == [0, 0, 0, 0]
    assert set(frame["Status"]) == {"placeholder"}


def test_dashboard_breakdown_frame_includes_live_stats_and_share_metrics():
    snapshot = RunSnapshot(
        run_id="run-123",
        stages={
            "energy": StageSnapshot(
                name="energy",
                status="completed",
                progress=1.0,
                detail="Energy ready",
            ),
            "vertices": StageSnapshot(
                name="vertices",
                status="completed",
                progress=1.0,
                detail="Vertices ready",
            ),
            "edges": StageSnapshot(
                name="edges",
                status="running",
                progress=0.5,
                detail="Tracing edges",
            ),
            "network": StageSnapshot(
                name="network",
                status="pending",
                progress=0.0,
            ),
        },
        optional_tasks={
            "share_report": TaskSnapshot(
                name="share_report",
                status="completed",
                progress=1.0,
                detail="Ready to download",
            )
        },
    )
    stats = {
        "num_strands": 12,
        "total_length": 345.6,
        "volume_fraction": 0.1234,
        "mean_radius": 4.56,
    }
    share_metrics = {
        "share_report_requested": 2,
        "share_report_downloaded": 1,
    }

    frame = web_app._dashboard_breakdown_frame(snapshot, stats, share_metrics)

    strands_row = frame.loc[frame["Metric"] == "Strands"].iloc[0]
    requested_row = frame.loc[frame["Metric"] == "Requested"].iloc[0]
    task_row = frame.loc[frame["Metric"] == "share_report"].iloc[0]

    assert strands_row["Value"] == "12"
    assert strands_row["Status"] == "live"
    assert requested_row["Value"] == "2"
    assert requested_row["Source"] == "session_state.share_report_metrics"
    assert task_row["Section"] == "Optional Tasks"
    assert task_row["Status"] == "completed"


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_render_export_download_uses_shared_success_path(monkeypatch):
    calls = {}

    monkeypatch.setattr(web_app, "generate_export_data", lambda *args, **kwargs: b"export-bytes")

    def fake_update(run_dir, task_name, **kwargs):
        calls["update"] = (run_dir, task_name, kwargs)

    def fake_download_button(**kwargs):
        calls["download"] = kwargs
        return False

    monkeypatch.setattr(web_app, "_update_run_task", fake_update)
    monkeypatch.setattr(web_app.st, "download_button", fake_download_button)
    monkeypatch.setattr(
        web_app.st,
        "button",
        lambda *args, **kwargs: calls.setdefault("button", (args, kwargs)),
    )

    web_app._render_export_download(
        _DummyColumn(),
        run_dir="run-dir",
        vertices={},
        edges={},
        network={},
        parameters={},
        export_spec=web_app.EXPORT_BUTTON_SPECS[0],
    )

    assert calls["update"][0] == "run-dir"
    assert calls["update"][1] == "exports"
    assert calls["update"][2]["artifacts"] == {"vmv_file": "network.vmv"}
    assert calls["download"]["file_name"] == "network.vmv"
    assert "button" not in calls


def test_render_export_download_uses_shared_failure_path(monkeypatch):
    calls = {}

    monkeypatch.setattr(web_app, "generate_export_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web_app,
        "_update_run_task",
        lambda *args, **kwargs: calls.setdefault("update", (args, kwargs)),
    )
    monkeypatch.setattr(
        web_app.st,
        "download_button",
        lambda **kwargs: calls.setdefault("download", kwargs),
    )

    def fake_button(*args, **kwargs):
        calls["button"] = (args, kwargs)
        return False

    monkeypatch.setattr(web_app.st, "button", fake_button)

    web_app._render_export_download(
        _DummyColumn(),
        run_dir="run-dir",
        vertices={},
        edges={},
        network={},
        parameters={},
        export_spec=web_app.EXPORT_BUTTON_SPECS[2],
    )

    assert calls["button"][0][0] == "📊 Export CSV"
    assert calls["button"][1]["disabled"] is True
    assert "update" not in calls
    assert "download" not in calls
