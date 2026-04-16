import sys
import types

import pytest

pytest.importorskip("streamlit")

from slavv.apps import web_app
from slavv.runtime.run_state import RunSnapshot, StageSnapshot, TaskSnapshot


def test_run_interactive_curator_dispatches_qt_backend(monkeypatch):
    fake_module = types.SimpleNamespace(
        run_curator=lambda energy, vertices, edges: ("qt-vertices", "qt-edges")
    )
    monkeypatch.setitem(sys.modules, "slavv.visualization.interactive_curator", fake_module)

    result = web_app._run_interactive_curator("energy", "vertices", "edges")

    assert result == ("qt-vertices", "qt-edges")


def test_run_interactive_curator_dispatches_napari_backend(monkeypatch):
    fake_module = types.SimpleNamespace(
        run_curator_napari=lambda energy, vertices, edges: ("napari-vertices", "napari-edges")
    )
    monkeypatch.setitem(sys.modules, "slavv.visualization.napari_curator", fake_module)

    result = web_app._run_interactive_curator("energy", "vertices", "edges", backend="napari")

    assert result == ("napari-vertices", "napari-edges")


def test_run_interactive_curator_rejects_unknown_backend():
    with pytest.raises(ValueError, match="curator backend"):
        web_app._run_interactive_curator("energy", "vertices", "edges", backend="unknown")


def test_dashboard_stage_frame_uses_placeholders_without_snapshot():
    frame = web_app._dashboard_stage_frame(None)

    assert frame["Stage"].tolist() == ["Energy", "Vertices", "Edges", "Network"]
    assert frame["Progress (%)"].tolist() == [0, 0, 0, 0]
    assert set(frame["Status"]) == {"placeholder"}
    assert set(frame["Source"]) == {"No run snapshot loaded"}


def test_dashboard_breakdown_frame_includes_live_stats_and_share_metrics():
    snapshot = RunSnapshot(
        run_id="run-123",
        status="running",
        elapsed_seconds=3665.0,
        eta_seconds=95.0,
        stages={
            "energy": StageSnapshot(
                name="energy",
                status="completed",
                progress=1.0,
                detail="Energy ready",
                resumed=True,
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
                resumed=True,
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
    runtime_row = frame.loc[frame["Metric"] == "Elapsed runtime"].iloc[0]
    eta_row = frame.loc[frame["Metric"] == "ETA"].iloc[0]
    resume_row = frame.loc[frame["Metric"] == "Resume rate"].iloc[0]

    assert strands_row["Value"] == "12"
    assert strands_row["Status"] == "live"
    assert requested_row["Value"] == "2"
    assert requested_row["Source"] == "session_state.share_report_metrics"
    assert task_row["Section"] == "Optional Tasks"
    assert task_row["Status"] == "completed"
    assert runtime_row["Value"] == "1h 1m"
    assert eta_row["Value"] == "1m 35.0s"
    assert resume_row["Value"] == "67% (2/3)"


def test_dashboard_breakdown_frame_uses_clean_empty_states_without_todo_labels():
    frame = web_app._dashboard_breakdown_frame(None, None, {}, run_dir=None)

    optional_row = frame.loc[frame["Metric"] == "Tracked tasks"].iloc[0]
    strands_row = frame.loc[frame["Metric"] == "Strands"].iloc[0]
    requested_row = frame.loc[frame["Metric"] == "Requested"].iloc[0]
    runtime_row = frame.loc[frame["Metric"] == "Elapsed runtime"].iloc[0]

    assert optional_row["Value"] == web_app.DASHBOARD_PLACEHOLDER
    assert optional_row["Source"] == "No run snapshot loaded"
    assert strands_row["Source"] == "session_state.processing_results"
    assert requested_row["Status"] == "idle"
    assert runtime_row["Value"] == web_app.DASHBOARD_PLACEHOLDER
    assert not frame["Value"].astype(str).str.contains("TODO").any()
    assert not frame["Source"].astype(str).str.contains("TODO").any()


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


def test_build_processing_run_dir_varies_with_validated_parameters():
    upload_bytes = b"same-uploaded-file"

    first = web_app._build_processing_run_dir(
        upload_bytes,
        {"radius_of_smallest_vessel_in_microns": 1.5},
    )
    second = web_app._build_processing_run_dir(
        upload_bytes,
        {"radius_of_smallest_vessel_in_microns": 2.0},
    )
    repeated = web_app._build_processing_run_dir(
        upload_bytes,
        {"radius_of_smallest_vessel_in_microns": 1.5},
    )

    assert first != second
    assert first == repeated
