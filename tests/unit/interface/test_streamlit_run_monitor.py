from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from slavv_python.engine.state.models import RunSnapshot, StageSnapshot
from slavv_python.interface.streamlit.services.run_monitor import (
    build_stage_unit_rows,
    format_age,
    heartbeat_age_seconds,
    infer_pipeline_route,
    load_run_ops_payload,
)


def _write_snapshot(root: Path, *, updated_at: str) -> None:
    metadata = root / "99_Metadata"
    metadata.mkdir(parents=True)
    (metadata / "run_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "status": "running",
                "target_stage": "network",
                "current_stage": "energy",
                "overall_progress": 0.1,
                "updated_at": updated_at,
                "stages": {},
                "provenance": {"slavv_python": "pipeline"},
            }
        ),
        encoding="utf-8",
    )


def test_infer_pipeline_route_detects_paper_and_parity(tmp_path: Path) -> None:
    paper = tmp_path / "paper-run"
    metadata = paper / "99_Metadata"
    metadata.mkdir(parents=True)
    (metadata / "validated_params.json").write_text(
        json.dumps({"edge_method": "tracing", "pipeline_profile": "paper"}),
        encoding="utf-8",
    )
    assert infer_pipeline_route(paper) == "Paper Path"

    parity = tmp_path / "parity-run"
    parity_meta = parity / "99_Metadata"
    parity_meta.mkdir(parents=True)
    (parity_meta / "validated_params.json").write_text(
        json.dumps({"edge_method": "tracing", "pipeline_profile": "matlab_compat"}),
        encoding="utf-8",
    )
    (parity_meta / "parity_job.json").write_text("{}", encoding="utf-8")
    assert infer_pipeline_route(parity) == "Exact Route (parity job)"

    exact = tmp_path / "exact-run"
    exact_meta = exact / "99_Metadata"
    exact_meta.mkdir(parents=True)
    (exact_meta / "validated_params.json").write_text(
        json.dumps({"edge_method": "watershed"}),
        encoding="utf-8",
    )
    assert infer_pipeline_route(exact) == "Exact Route"


def test_heartbeat_age_uses_resume_state_and_snapshot(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    root = tmp_path / "run"
    energy = root / "02_Energy"
    energy.mkdir(parents=True)
    (energy / "resume_state.json").write_text(
        json.dumps({"heartbeat_at": fresh}),
        encoding="utf-8",
    )
    _write_snapshot(root, updated_at=stale)
    from slavv_python.engine.state import load_run_snapshot

    snapshot = load_run_snapshot(root)
    age = heartbeat_age_seconds(root, snapshot)
    assert age is not None
    assert age < 90
    assert format_age(age).endswith("s ago")


def test_load_run_ops_payload_includes_log_tail(tmp_path: Path) -> None:
    root = tmp_path / "logged-run"
    _write_snapshot(
        root,
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    log_path = root / "99_Metadata" / "parity_job.out.log"
    log_path.write_text(
        "warmup\nchunk 1\nchunk 2\nchunk 3\nchunk 4\nchunk 5\nchunk 6\n",
        encoding="utf-8",
    )
    payload = load_run_ops_payload(root, max_log_lines=5)
    assert payload["log_tail"]["name"] == "parity_job.out.log"
    assert list(payload["log_tail"]["lines"]) == [
        "chunk 2",
        "chunk 3",
        "chunk 4",
        "chunk 5",
        "chunk 6",
    ]
    assert payload["effective_status"]


def test_build_stage_unit_rows_reports_completed_over_total() -> None:
    snapshot = RunSnapshot(
        run_id="units",
        current_stage="energy",
        stages={
            "energy": StageSnapshot(
                name="energy",
                status="running",
                progress=0.25,
                units_completed=50,
                units_total=200,
            ),
            "vertices": StageSnapshot(name="vertices", status="pending"),
        },
    )
    rows = {row["stage"]: row for row in build_stage_unit_rows(snapshot)}
    assert rows["energy"]["units_label"] == "50/200"
    assert rows["energy"]["fraction"] == pytest.approx(0.25)
    assert rows["vertices"]["units_label"] == "—"
    assert rows["vertices"]["fraction"] == 0.0
