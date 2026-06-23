from __future__ import annotations

import json

import pytest

from slavv_python.analytics.parity.job_registry import JobRegistry
from slavv_python.analytics.parity.parity_job_lifecycle import (
    finalize_parity_job,
    load_parity_job_manifest,
    mark_parity_job_running,
    parity_job_manifest_path,
    reconcile_interrupted_run,
)
from slavv_python.analytics.parity.writer_lease import load_writer_lease, write_writer_lease
from slavv_python.interface.cli.monitor_service import load_run_monitor_view
from tests.support.run_state_builders import (
    build_snapshot_dict,
    build_stage_snapshot_dict,
    materialize_run_snapshot,
)


@pytest.mark.unit
def test_finalize_parity_job_persists_terminal_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "slavv_python.analytics.parity.parity_job_lifecycle.now_iso",
        lambda: "2026-06-22T12:00:00Z",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    manifest = finalize_parity_job(
        run_dir,
        status="interrupted",
        exit_code=None,
        reason="Snapshot is running but PID 12345 is dead.",
    )

    assert manifest["status"] == "interrupted"
    assert manifest["ended_at"] == "2026-06-22T12:00:00Z"
    assert manifest["exit_code"] is None
    assert manifest["reason"] == "Snapshot is running but PID 12345 is dead."
    assert parity_job_manifest_path(run_dir).with_name("parity_job.json.sha256").is_file()


@pytest.mark.unit
def test_reconcile_interrupted_run_finalizes_writer_lease_and_registry(tmp_path, monkeypatch):
    frozen_now = "2026-06-22T12:00:00Z"
    monkeypatch.setattr(
        "slavv_python.analytics.parity.parity_job_lifecycle.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.writer_lease.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.writer_lease.resolve_python_commit",
        lambda _root: "abc123",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_writer_lease(
        run_dir,
        pid=12345,
        command="resume-exact-run",
        stage="energy",
        status="running",
    )
    registry = JobRegistry(tmp_path / "registry.jsonl")
    monkeypatch.setattr(
        "slavv_python.analytics.parity.job_registry.JobRegistry",
        lambda registry_path=None: registry,
    )
    job_id = registry.register_job(
        pid=12345,
        run_dir=run_dir,
        oracle_root=tmp_path / "oracle",
        stage="energy",
        command="resume-exact-run",
    )

    manifest = reconcile_interrupted_run(
        run_dir,
        reason="Snapshot is running but PID 12345 is dead.",
    )

    assert manifest is not None
    assert manifest["status"] == "interrupted"
    assert load_writer_lease(run_dir)["status"] == "interrupted"
    assert load_writer_lease(run_dir)["ended_at"] == "2026-06-22T12:00:00Z"
    record = registry.get_job_by_run_dir(run_dir)
    assert record is not None
    assert record.job_id == job_id
    assert record.status == "interrupted"


@pytest.mark.unit
def test_monitor_view_reconciles_dead_pid_to_interrupted(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(
        run_dir,
        build_snapshot_dict(
            status="running",
            current_stage="energy",
            stages={"energy": build_stage_snapshot_dict("energy", status="running")},
        ),
    )
    metadata = run_dir / "99_Metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "parity_job.pid").write_text("12345", encoding="utf-8")
    (metadata / "parity_job.json").write_text(
        json.dumps({"kind": "parity_exact_run_job", "status": "running", "pid": 12345}),
        encoding="utf-8",
    )
    write_writer_lease(
        run_dir,
        pid=12345,
        command="resume-exact-run",
        stage="energy",
        status="running",
    )
    monkeypatch.setattr(
        "slavv_python.interface.cli.monitor_service._process_command_line",
        lambda pid: None,
    )

    view = load_run_monitor_view(run_dir)

    assert view.effective_status == "interrupted"
    manifest = load_parity_job_manifest(run_dir)
    assert manifest is not None
    assert manifest["status"] == "interrupted"
    assert manifest["ended_at"]
    assert load_writer_lease(run_dir)["status"] == "interrupted"


@pytest.mark.unit
def test_mark_parity_job_running_preserves_started_at(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "slavv_python.analytics.parity.parity_job_lifecycle.now_iso",
        lambda: "2026-06-22T13:00:00Z",
    )
    run_dir = tmp_path / "run"
    metadata = run_dir / "99_Metadata"
    metadata.mkdir(parents=True)
    parity_job_manifest_path(run_dir).write_text(
        json.dumps({"started_at": "2026-06-22T10:00:00Z", "status": "running"}),
        encoding="utf-8",
    )

    manifest = mark_parity_job_running(
        run_dir,
        pid=999,
        command=["python", "-m", "parity", "resume-exact-run"],
        stage="energy",
    )

    assert manifest["started_at"] == "2026-06-22T10:00:00Z"
    assert manifest["status"] == "running"
    assert manifest["pid"] == 999
