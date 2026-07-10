from __future__ import annotations

import pytest

from slavv_python.analytics.parity.runs.job_registry import JobRegistry
from slavv_python.analytics.parity.runs.parity_job_lifecycle import (
    load_parity_job_manifest,
    parity_job_manifest_path,
)
from slavv_python.analytics.parity.runs.writer_lease import (
    load_writer_lease,
    write_writer_lease,
)
from slavv_python.analytics.parity.runs.writer_session import (
    launch_writer_session,
    reconcile_stale_writer_lease,
    resume_writer_session,
)


@pytest.mark.unit
def test_reconcile_stale_writer_lease_clears_dead_pid(tmp_path, monkeypatch):
    frozen_now = "2026-07-04T12:00:00Z"
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.now_iso",
        lambda: frozen_now,
        raising=False,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.parity_job_lifecycle.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.resolve_python_commit",
        lambda _root: "abc123",
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.is_process_alive",
        lambda _pid: False,
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

    reconcile_stale_writer_lease(run_dir)

    lease = load_writer_lease(run_dir)
    assert lease is not None
    assert lease["status"] == "interrupted"
    manifest = load_parity_job_manifest(run_dir)
    assert manifest is not None
    assert manifest["status"] == "interrupted"
    assert manifest["reason"] == "Writer lease PID 12345 is no longer alive."


@pytest.mark.unit
def test_resume_writer_session_success_finalizes_lease_and_manifest(tmp_path, monkeypatch):
    frozen_now = "2026-07-04T13:00:00Z"
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.parity_job_lifecycle.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.resolve_python_commit",
        lambda _root: "abc123",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parity_job_manifest_path(run_dir).parent.mkdir(parents=True, exist_ok=True)

    with resume_writer_session(
        run_dir,
        command="resume-exact-run",
        stage="energy",
        argv=["slavv", "parity", "resume-exact-run"],
        stop_after="vertices",
    ):
        pass

    lease = load_writer_lease(run_dir)
    assert lease is not None
    assert lease["status"] == "completed"
    assert lease["stage"] == "vertices"
    manifest = load_parity_job_manifest(run_dir)
    assert manifest is not None
    assert manifest["status"] == "succeeded"
    assert manifest["exit_code"] == 0


@pytest.mark.unit
def test_resume_writer_session_failure_finalizes_failed(tmp_path, monkeypatch):
    frozen_now = "2026-07-04T14:00:00Z"
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.parity_job_lifecycle.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.resolve_python_commit",
        lambda _root: "abc123",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parity_job_manifest_path(run_dir).parent.mkdir(parents=True, exist_ok=True)

    with (
        pytest.raises(RuntimeError, match="boom"),
        resume_writer_session(
            run_dir,
            command="resume-exact-run",
            stage="energy",
            argv=["slavv", "parity", "resume-exact-run"],
        ),
    ):
        raise RuntimeError("boom")

    lease = load_writer_lease(run_dir)
    assert lease is not None
    assert lease["status"] == "failed"
    manifest = load_parity_job_manifest(run_dir)
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 1
    assert manifest["reason"] == "boom"


@pytest.mark.unit
def test_resume_writer_session_monitor_registers_and_succeeds(tmp_path, monkeypatch):
    frozen_now = "2026-07-04T15:00:00Z"
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.parity_job_lifecycle.now_iso",
        lambda: frozen_now,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.resolve_python_commit",
        lambda _root: "abc123",
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.ensure_monitor_daemon_running",
        lambda: None,
    )
    registry = JobRegistry(tmp_path / "registry.jsonl")
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.JobRegistry",
        lambda registry_path=None: registry,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()

    with resume_writer_session(
        run_dir,
        command="resume-exact-run",
        stage="energy",
        argv=["slavv", "parity", "resume-exact-run"],
        monitor=True,
        oracle_root=oracle_root,
    ):
        pass

    record = registry.get_job_by_run_dir(run_dir)
    assert record is not None
    assert record.status == "succeeded"
    assert record.exit_code == 0


@pytest.mark.unit
def test_launch_writer_session_prepare_spawn_and_monitor(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    registry = JobRegistry(tmp_path / "registry.jsonl")
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.JobRegistry",
        lambda registry_path=None: registry,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.ensure_monitor_daemon_running",
        lambda: None,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.reconcile_stale_writer_lease",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.reconcile_registry_writer_conflict",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.prepare_detached_exact_run_launch",
        lambda **_kwargs: (["python", "-m", "detached"], ["python", "-m", "probe"]),
    )

    def _fake_launch(**kwargs):
        assert kwargs["command_override"] == ["python", "-m", "detached"]
        assert kwargs["skip_preflight"] is True
        return {
            "pid": 4242,
            "stdout": str(run_dir / "out.log"),
            "stderr": str(run_dir / "err.log"),
            "command": ["python", "-m", "detached"],
            "oracle_root": str(oracle_root),
            "pid_file": str(run_dir / "job.pid"),
        }

    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_session.launch_exact_run_job",
        _fake_launch,
    )

    manifest = launch_writer_session(
        run_dir,
        oracle_root=oracle_root,
        stop_after="edges",
        force_rerun_from="edges",
        skip_preflight=True,
        skip_foreground_probe=True,
        monitor=True,
    )

    assert manifest["pid"] == 4242
    record = registry.get_job_by_run_dir(run_dir)
    assert record is not None
    assert record.pid == 4242
    assert record.stage == "edges"
