"""Tests for file-backed resumable run state."""

from __future__ import annotations

import json

import joblib
import pytest

from slavv.runtime import RunContext, load_run_snapshot
from slavv.runtime.run_state import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_PENDING,
    atomic_write_json,
    target_stage_progress,
)


def test_atomic_write_json_replaces_previous_content(tmp_path):
    path = tmp_path / "snapshot.json"

    atomic_write_json(path, {"stage": "energy", "progress": 0.25})
    atomic_write_json(path, {"stage": "network", "progress": 1.0})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "progress": 1.0,
        "stage": "network",
    }


def test_run_context_persists_snapshot_lifecycle(tmp_path):
    run_dir = tmp_path / "run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="edges",
        provenance={"source": "unit-test"},
    )

    context.mark_preprocess_complete()
    energy_stage = context.stage("energy")
    energy_stage.begin(detail="Computing energy", units_total=4, units_completed=1)
    energy_stage.update(units_completed=2, detail="Halfway through energy")
    energy_stage.save_checkpoint({"energy": "ready"})
    energy_stage.complete(detail="Energy complete", artifacts={"energy_field": "energy.npy"})

    context.update_optional_task("exports", status="running", detail="Preparing exports")
    context.update_optional_task(
        "exports",
        status="completed",
        detail="Exported requested outputs",
        artifacts={"json": "network.json"},
    )
    context.finalize_run(stop_after="edges")

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_COMPLETED_TARGET
    assert snapshot.target_stage == "edges"
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].artifacts["checkpoint"].endswith("checkpoint_energy.pkl")
    assert snapshot.stages["energy"].artifacts["energy_field"] == "energy.npy"
    assert snapshot.optional_tasks["exports"].status == STATUS_COMPLETED
    assert snapshot.optional_tasks["exports"].artifacts["json"] == "network.json"
    assert target_stage_progress(snapshot) >= snapshot.overall_progress


def test_resume_guard_blocks_mismatched_input(tmp_path):
    run_dir = tmp_path / "blocked_run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )

    with pytest.raises(RuntimeError, match="Resume blocked"):
        context.ensure_resume_allowed(
            input_fingerprint="input-b",
            params_fingerprint="params-a",
        )

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_BLOCKED
    assert "fingerprint changed" in snapshot.last_event
    assert snapshot.errors


def test_force_rerun_from_energy_resets_pipeline_state(tmp_path):
    run_dir = tmp_path / "reset_run"
    context = RunContext(
        run_dir=run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )
    energy_stage = context.stage("energy")
    energy_stage.save_checkpoint({"energy": "ready"})
    energy_stage.complete(detail="Energy complete")

    assert energy_stage.checkpoint_path.exists()

    context.ensure_resume_allowed(
        input_fingerprint="input-b",
        params_fingerprint="params-a",
        force_rerun_from="energy",
    )

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_PENDING
    assert snapshot.input_fingerprint == "input-b"
    assert snapshot.stages["energy"].status == STATUS_PENDING
    assert not energy_stage.checkpoint_path.exists()


def test_legacy_checkpoints_bootstrap_snapshot(tmp_path):
    checkpoint_dir = tmp_path / "legacy_checkpoints"
    checkpoint_dir.mkdir()
    joblib.dump({"energy": True}, checkpoint_dir / "checkpoint_energy.pkl")
    joblib.dump({"vertices": True}, checkpoint_dir / "checkpoint_vertices.pkl")

    context = RunContext(
        checkpoint_dir=checkpoint_dir,
        target_stage="network",
        legacy=True,
    )

    snapshot = context.snapshot
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["vertices"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].resumed is True
    assert snapshot.stages["edges"].status == STATUS_PENDING
    assert load_run_snapshot(checkpoint_dir) is not None
