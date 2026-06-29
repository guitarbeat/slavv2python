from __future__ import annotations

import json

from slavv_python.interface.cli import monitor_service
from slavv_python.interface.cli.monitor_service import load_run_monitor_view, render_monitor_lines
from tests.support.run_state_builders import (
    build_snapshot_dict,
    build_stage_snapshot_dict,
    materialize_run_snapshot,
)


def test_monitor_view_reports_stale_running_snapshot_when_pid_is_dead(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(
        run_dir,
        build_snapshot_dict(
            status="running",
            current_stage="energy",
            stages={"energy": build_stage_snapshot_dict("energy", status="running")},
        ),
    )
    pid_path = run_dir / "99_Metadata" / "run.pid"
    pid_path.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(monitor_service, "_process_command_line", lambda pid: None)

    view = load_run_monitor_view(run_dir)

    assert view.effective_status == "interrupted"
    assert view.pid_statuses[0].state == "dead"


def test_monitor_view_reports_live_pid_for_matching_command_line(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(run_dir, build_snapshot_dict(status="running"))
    pid_path = run_dir / "99_Metadata" / "run.pid"
    pid_path.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(
        monitor_service,
        "_process_command_line",
        lambda pid: f"python parity_experiment.py --dest-run-root {run_dir}",
    )

    view = load_run_monitor_view(run_dir)

    assert view.effective_status == "running"
    assert view.pid_statuses[0].state == "alive"


def test_monitor_view_discovers_parity_job_pid(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(run_dir, build_snapshot_dict(status="running"))
    pid_path = run_dir / "99_Metadata" / "parity_job.pid"
    pid_path.write_text("98765", encoding="utf-8")
    monkeypatch.setattr(
        monitor_service,
        "_process_command_line",
        lambda pid: (
            f"python scripts/parity_experiment.py resume-exact-run --dest-run-root {run_dir}"
        ),
    )

    view = load_run_monitor_view(run_dir)

    assert view.effective_status == "running"
    assert any(status.path == pid_path for status in view.pid_statuses)


def test_monitor_view_reports_conflicting_live_lease_and_registry(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(run_dir, build_snapshot_dict(status="running"))
    lease_path = run_dir / "99_Metadata" / "writer_lease.json"
    lease_path.write_text(json.dumps({"pid": 12345}), encoding="utf-8")

    class Record:
        pid = 98765
        status = "running"
        command = "python parity"

    class Registry:
        def get_job_by_run_dir(self, _run_dir):
            return Record()

    monkeypatch.setattr(monitor_service, "_process_command_line", lambda _pid: "python parity")
    monkeypatch.setattr("slavv_python.analytics.parity.runs.job_registry.JobRegistry", Registry)

    view = load_run_monitor_view(run_dir)

    assert view.effective_status == "conflicting-writers"


def test_monitor_view_summarizes_parity_proof_json(tmp_path):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(run_dir, build_snapshot_dict(status="completed_target"))
    analysis = run_dir / "03_Analysis"
    analysis.mkdir(parents=True)
    (analysis / "exact_proof_energy.json").write_text(
        json.dumps(
            {
                "passed": False,
                "first_failing_stage": "energy",
                "first_failing_field_path": "energy.energy",
            }
        ),
        encoding="utf-8",
    )

    view = load_run_monitor_view(run_dir)
    lines = render_monitor_lines(view)

    assert view.proof_statuses[0].passed is False
    assert any("exact_proof_energy.json: failed" in line for line in lines)


def test_monitor_view_missing_snapshot_is_actionable(tmp_path):
    view = load_run_monitor_view(tmp_path / "missing")

    assert view.effective_status == "missing-snapshot"
    assert view.errors
    assert "Missing snapshot" in view.errors[0]
