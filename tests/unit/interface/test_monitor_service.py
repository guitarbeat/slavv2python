from __future__ import annotations

import json
import os

import pytest

from slavv_python.engine.constants import TRACKED_RUN_STAGES
from slavv_python.interface.cli import monitor_service
from slavv_python.interface.cli.monitor_service import (
    EnergyProgress,
    build_stage_rows,
    compute_energy_progress,
    format_duration,
    live_overall_progress,
    load_run_monitor_view,
    render_monitor_lines,
    status_style,
    tail_log_lines,
)
from tests.support.run_state_builders import (
    build_snapshot_dict,
    build_stage_snapshot_dict,
    materialize_run_snapshot,
)

_OLD_TS = "2020-01-01T00:00:00Z"
_JOBLIB_LOG = (
    "[Parallel(n_jobs=6)]: Done  20 tasks      | elapsed:   20.0s\n"
    "[Parallel(n_jobs=6)]: Done  40 tasks      | elapsed:   40.0s\n"
)


def _materialize_energy_run(
    run_dir,
    *,
    units_total: int = 512,
    units_completed: int = 100,
    stage_updated_at: str = _OLD_TS,
    current_stage: str = "energy",
    log_body: str | None = _JOBLIB_LOG,
    preprocess_done: bool = False,
):
    """Write an energy-stage snapshot (+ optional joblib log) under ``run_dir``."""
    energy_stage = build_stage_snapshot_dict(
        "energy",
        status="running",
        units_total=units_total,
        units_completed=units_completed,
    )
    energy_stage["updated_at"] = stage_updated_at
    artifacts = {"preprocess_done": "true"} if preprocess_done else None
    materialize_run_snapshot(
        run_dir,
        build_snapshot_dict(
            status="running",
            current_stage=current_stage,
            stages={"energy": energy_stage},
            artifacts=artifacts,
        ),
    )
    log_path = run_dir / "99_Metadata" / "parity_job.out.log"
    if log_body is not None:
        log_path.write_text(log_body, encoding="utf-8")
    return log_path


def test_compute_energy_progress_augments_with_fresh_joblib_log(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=100)

    energy = compute_energy_progress(load_run_monitor_view(run_dir))

    assert energy is not None
    assert energy.durable_units_completed == 100
    assert energy.live_units_completed == 140  # 100 durable + 40 chunks in flight
    assert energy.is_live is True
    assert energy.per_chunk_seconds == 1.0  # (40-20)s / (40-20) chunks
    assert energy.eta_seconds == 372.0  # (512 - 140) chunks * 1.0s


def test_compute_energy_progress_ignores_stale_log(tmp_path):
    run_dir = tmp_path / "run"
    # Durable checkpoint is newer than the log -> log must not be trusted.
    log_path = _materialize_energy_run(
        run_dir, units_completed=100, stage_updated_at="2099-01-01T00:00:00Z"
    )
    os.utime(log_path, (0, 0))

    energy = compute_energy_progress(load_run_monitor_view(run_dir))

    assert energy is not None
    assert energy.live_units_completed == 100
    assert energy.is_live is False


def test_compute_energy_progress_caps_live_units_at_total(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=500)

    energy = compute_energy_progress(load_run_monitor_view(run_dir))

    assert energy is not None
    assert energy.live_units_completed == 512  # min(500 + 40, 512)
    assert energy.fraction == 1.0


def test_compute_energy_progress_none_when_not_energy_stage(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, current_stage="vertices")

    assert compute_energy_progress(load_run_monitor_view(run_dir)) is None


def test_compute_energy_progress_uses_durable_without_log(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_completed=140, log_body=None)

    energy = compute_energy_progress(load_run_monitor_view(run_dir))

    assert energy is not None
    assert energy.live_units_completed == 140
    assert energy.is_live is False


def test_render_monitor_lines_includes_live_energy_line(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=100)

    lines = render_monitor_lines(load_run_monitor_view(run_dir))

    assert any("Energy chunks (live from log): 140/512" in line for line in lines)


def test_compute_energy_progress_rate_uses_trailing_batch_after_reset(tmp_path):
    run_dir = tmp_path / "run"
    # A stale prior-octave batch precedes the current one; joblib resets Done N.
    log_body = (
        "[Parallel(n_jobs=6)]: Done 100 tasks | elapsed:  200.0s\n"
        "[Parallel(n_jobs=6)]: Done  10 tasks | elapsed:   10.0s\n"
        "[Parallel(n_jobs=6)]: Done  30 tasks | elapsed:   40.0s\n"
    )
    _materialize_energy_run(run_dir, units_total=512, units_completed=100, log_body=log_body)

    energy = compute_energy_progress(load_run_monitor_view(run_dir))

    assert energy is not None
    assert energy.chunks_in_batch == 30  # trailing batch, not the stale 100
    assert energy.live_units_completed == 130
    assert energy.per_chunk_seconds == 1.5  # (40-10)s / (30-10) chunks


def test_live_overall_progress_advances_with_live_energy_fraction(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=100, preprocess_done=True)
    view = load_run_monitor_view(run_dir)
    energy = compute_energy_progress(view)

    assert energy is not None
    live_overall = live_overall_progress(view.snapshot, energy)
    # preprocess weight (0.05) + energy weight (0.35) * live fraction (140/512).
    assert live_overall == pytest.approx(0.05 + 0.35 * (140 / 512))
    assert live_overall > view.snapshot.overall_progress


def test_render_monitor_lines_honors_precomputed_energy(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=100)
    view = load_run_monitor_view(run_dir)

    injected = EnergyProgress(
        units_total=1000,
        durable_units_completed=1,
        live_units_completed=999,
        chunks_in_batch=998,
        per_chunk_seconds=None,
        eta_seconds=None,
        is_live=True,
        log_path=None,
    )
    lines = render_monitor_lines(view, energy=injected)
    assert any("Energy chunks (live from log): 999/1000" in line for line in lines)

    # Passing None suppresses the energy line even though the log exists.
    none_lines = render_monitor_lines(view, energy=None)
    assert not any("Energy chunks" in line for line in none_lines)


def test_render_monitor_lines_lists_full_pipeline_with_current_marker(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=100)

    lines = render_monitor_lines(load_run_monitor_view(run_dir))

    # Every tracked stage appears, even ones absent from the snapshot.
    for stage in TRACKED_RUN_STAGES:
        assert any(f" {stage}: " in line for line in lines)
    # The current stage is marked, and not-yet-started stages read as pending.
    assert any(line.strip().startswith("> energy:") for line in lines)
    assert any(line.strip().startswith("- network: pending") for line in lines)


def test_build_stage_rows_covers_every_tracked_stage_in_order(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, units_total=512, units_completed=100)
    snapshot = load_run_monitor_view(run_dir).snapshot

    rows = build_stage_rows(snapshot)

    assert [row.name for row in rows] == list(TRACKED_RUN_STAGES)
    energy_row = next(row for row in rows if row.name == "energy")
    assert energy_row.status == "running"
    assert energy_row.units_label == "100/512"
    # Stages absent from the snapshot are reported as pending with no units.
    vertices_row = next(row for row in rows if row.name == "vertices")
    assert vertices_row.status == "pending"
    assert vertices_row.units_label == "—"


def test_status_style_maps_status_keywords():
    assert status_style("running") == "green"
    assert status_style("failed") == "red"
    assert status_style("resume_blocked") == "red"
    assert status_style("interrupted") == "yellow"
    assert status_style("stale-running-snapshot") == "yellow"
    assert status_style("completed_target") == "cyan"
    assert status_style("pending") == "grey62"
    assert status_style("something-else") == "white"


def test_tail_log_lines_returns_newest_log(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, log_body="line-a\nline-b\n\nline-c\n")

    name, lines = tail_log_lines(load_run_monitor_view(run_dir), max_lines=2)

    assert name == "parity_job.out.log"
    assert lines == ["line-b", "line-c"]  # blank line dropped, capped to last 2


def test_tail_log_lines_empty_without_logs(tmp_path):
    run_dir = tmp_path / "run"
    _materialize_energy_run(run_dir, log_body=None)

    name, lines = tail_log_lines(load_run_monitor_view(run_dir))

    assert name is None
    assert lines == []


def test_format_duration_uses_largest_unit():
    assert format_duration(30) == "~30s"
    assert format_duration(90) == "~1.5m"
    assert format_duration(7200) == "~2.0h"


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

    energy_proof = next(p for p in view.proof_statuses if p.path.name == "exact_proof_energy.json")
    assert energy_proof.passed is False
    assert any("exact_proof_energy.json: failed" in line for line in lines)


def test_monitor_recovers_pipeline_stages_from_parity_manifest(tmp_path):
    run_dir = tmp_path / "canonical_full_v5"
    metadata = run_dir / "99_Metadata"
    metadata.mkdir(parents=True)
    (metadata / "run_snapshot.json").write_text(
        json.dumps(
            {
                "kind": "parity_run",
                "run_id": "808667c39ad8",
                "status": "failed",
                "stage_metrics": {
                    "energy": {"status": "completed", "elapsed_seconds": 0.0},
                    "vertices": {"status": "completed", "elapsed_seconds": 0.0},
                    "edges": {"status": "completed", "elapsed_seconds": 7360.0},
                    "network": {"status": "completed", "elapsed_seconds": 300.0},
                },
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "run_manifest.json").write_text(
        (metadata / "run_snapshot.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (metadata / "parity_job.json").write_text(
        json.dumps({"status": "succeeded", "stop_after": "network"}),
        encoding="utf-8",
    )

    view = load_run_monitor_view(run_dir)

    assert view.snapshot is not None
    assert view.snapshot.status == "completed_target"
    assert view.snapshot.stages["edges"].status == "completed"
    assert view.snapshot.stages["network"].status == "completed"
    assert view.effective_status == "succeeded"
    lines = render_monitor_lines(view)
    assert any("edges: completed" in line for line in lines)
    assert any("network: completed" in line for line in lines)


def test_monitor_view_missing_snapshot_is_actionable(tmp_path):
    view = load_run_monitor_view(tmp_path / "missing")

    assert view.effective_status == "missing-snapshot"
    assert view.errors
    assert "Missing snapshot" in view.errors[0]
