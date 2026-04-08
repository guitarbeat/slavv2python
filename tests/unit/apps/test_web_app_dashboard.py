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
