"""Regression coverage for exact Energy proof evidence freshness."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import pytest

from slavv_python.analytics.parity.cli_handlers.cli_diagnostics import handle_inspect_energy_evidence
from slavv_python.analytics.parity.oracle.models import ExactProofSourceSurface, OracleSurface
from slavv_python.analytics.parity.proof.coordinator import ExactProofCoordinator
from slavv_python.analytics.parity.proof.energy_proof_evidence import (
    build_energy_proof_evidence,
    require_energy_proof_evidence,
)
from tests.support.batch_energy_mismatch_probe import main as batch_energy_mismatch_probe_main


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot(*, status: str, started_at: str, updated_at: str) -> dict[str, object]:
    return {
        "status": status,
        "stages": {
            "energy": {
                "status": status,
                "started_at": started_at,
                "updated_at": updated_at,
            }
        },
    }


def _energy_checkpoint(run_root, timestamp: datetime):
    path = run_root / "02_Output" / "python_results" / "checkpoints" / "checkpoint_energy.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    epoch = timestamp.timestamp()
    os.utime(path, (epoch, epoch))
    return path


def _source_surface(run_root):
    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / "02_Output" / "python_results" / "checkpoints",
        validated_params_path=run_root / "01_Params" / "validated_params.json",
        oracle_surface=OracleSurface(
            oracle_root=None,
            manifest_path=None,
            matlab_batch_dir=None,
            matlab_vector_paths={},
            oracle_id="test-oracle",
            matlab_source_version=None,
            dataset_hash=None,
        ),
        matlab_batch_dir=None,
        matlab_vector_paths={},
    )


def test_evidence_accepts_completed_energy_from_stage_metrics_when_stages_empty(tmp_path):
    started = datetime(2026, 6, 24, 1, 33, 9, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        {
            "status": "failed",
            "stages": {},
            "stage_metrics": {
                "energy": {
                    "status": "completed",
                    "completed_at": "2026-06-24T09:25:23Z",
                    "elapsed_seconds": 28332.0,
                    "peak_memory_bytes": 302534656,
                }
            },
        },
    )
    _write_json(
        tmp_path / "99_Metadata" / "writer_lease.json",
        {
            "stage": "energy",
            "status": "completed",
            "started_at": "2026-06-24T01:33:09Z",
            "updated_at": "2026-06-24T09:25:23Z",
        },
    )
    _energy_checkpoint(tmp_path, started + timedelta(hours=8))

    report = build_energy_proof_evidence(tmp_path)

    assert report["valid"] is True
    assert report["failures"] == []
    assert report["run_snapshot"]["energy_status"] == "completed"
    assert report["run_snapshot"]["energy_started_at"] == "2026-06-24T01:33:09Z"


def test_evidence_accepts_completed_energy_checkpoint_newer_than_latest_start(tmp_path):
    started = datetime(2026, 6, 23, 1, 0, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="completed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )
    _energy_checkpoint(tmp_path, started + timedelta(seconds=1))

    report = build_energy_proof_evidence(tmp_path)

    assert report["valid"] is True
    assert report["failures"] == []


def test_evidence_rejects_missing_energy_checkpoint_and_failed_snapshot(tmp_path):
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="failed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )
    stale_report = tmp_path / "03_Analysis" / "exact_proof_energy.json"
    _write_json(stale_report, {"passed": False})

    report = build_energy_proof_evidence(tmp_path)

    assert report["valid"] is False
    assert "missing_energy_checkpoint" in report["failures"]
    assert "energy_stage_status_failed" in report["failures"]
    assert report["historical_reports"][0]["stale"] is True
    with pytest.raises(ValueError, match="Energy proof evidence is stale"):
        require_energy_proof_evidence(tmp_path)


def test_evidence_rejects_checkpoint_older_than_latest_energy_attempt(tmp_path):
    started = datetime(2026, 6, 23, 1, 0, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="completed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )
    _energy_checkpoint(tmp_path, started - timedelta(seconds=1))

    report = build_energy_proof_evidence(tmp_path)

    assert report["valid"] is False
    assert "energy_checkpoint_predates_latest_start" in report["failures"]


def test_evidence_marks_reports_older_than_current_checkpoint_stale(tmp_path):
    started = datetime(2026, 6, 23, 1, 0, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="completed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )
    report_path = tmp_path / "03_Analysis" / "exact_mismatch_energy.json"
    _write_json(report_path, {"mismatch_count": 1})
    report_epoch = (started - timedelta(seconds=1)).timestamp()
    os.utime(report_path, (report_epoch, report_epoch))
    _energy_checkpoint(tmp_path, started + timedelta(seconds=1))

    report = build_energy_proof_evidence(tmp_path)

    assert report["valid"] is True
    assert report["historical_reports"] == [
        {
            "path": str(report_path),
            "stale": True,
            "stale_reason": "predates_current_energy_checkpoint",
        }
    ]


def test_inspect_energy_evidence_writes_report_before_failing_stale_run(tmp_path):
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="failed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )
    output = tmp_path / "03_Analysis" / "custom_energy_evidence.json"

    with pytest.raises(SystemExit) as excinfo:
        handle_inspect_energy_evidence(Namespace(run_root=tmp_path, output=output))

    assert excinfo.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert "missing_energy_checkpoint" in report["failures"]
    assert output.with_name(f"{output.name}.sha256").is_file()


def test_batch_mismatch_probe_rejects_stale_energy_evidence_before_export(tmp_path):
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="failed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )
    probe_requests = tmp_path / "03_Analysis" / "energy_probe_requests.json"
    _write_json(probe_requests, {"requests": []})

    with pytest.raises(ValueError, match="Energy proof evidence is stale"):
        batch_energy_mismatch_probe_main(
            [
                "--mode",
                "export-requests",
                "--probe-requests",
                str(probe_requests),
                "--output",
                str(tmp_path / "03_Analysis" / "batch_requests.json"),
            ]
        )


@pytest.mark.parametrize("stage_arg", ["energy", "all"])
def test_prove_exact_rejects_stale_energy_evidence_before_loading_artifacts(
    tmp_path, monkeypatch, stage_arg
):
    _write_json(
        tmp_path / "99_Metadata" / "run_snapshot.json",
        _snapshot(
            status="failed",
            started_at="2026-06-23T01:00:00Z",
            updated_at="2026-06-23T01:02:00Z",
        ),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("artifact loading must not run for stale Energy evidence")

    monkeypatch.setattr(
        "slavv_python.analytics.parity.proof.coordinator.load_normalized_matlab_vectors",
        fail_if_called,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.proof.coordinator.load_normalized_python_checkpoints",
        fail_if_called,
    )

    coordinator = ExactProofCoordinator(_source_surface(tmp_path))

    with pytest.raises(ValueError, match="Energy proof evidence is stale"):
        coordinator.prove(tmp_path, stage_arg=stage_arg)


def test_prove_exact_non_energy_stage_is_not_blocked_by_energy_evidence(tmp_path, monkeypatch):
    coordinator = ExactProofCoordinator(_source_surface(tmp_path))

    monkeypatch.setattr(
        "slavv_python.analytics.parity.proof.coordinator.load_normalized_python_checkpoints",
        lambda _path, stages: {stage: {} for stage in stages},
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.proof.coordinator.compare_exact_artifacts",
        lambda _matlab, _python, stages, **_kwargs: {"passed": True, "stages": list(stages)},
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.proof.coordinator.render_exact_proof_report",
        lambda _report: "report",
    )
    monkeypatch.setattr(
        ExactProofCoordinator,
        "_write_release_evidence",
        lambda self, dest_run_root, report_payload, params: None,
    )

    report, _, _ = coordinator.prove(tmp_path, stage_arg="vertices")

    assert report["passed"] is True
