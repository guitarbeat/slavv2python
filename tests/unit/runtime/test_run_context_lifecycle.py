"""Tests for structured run-context lifecycle behavior."""

from __future__ import annotations

import json

import pytest
from tests.support.run_state_builders import build_run_context

from slavv_python.runtime import RunContext, load_run_snapshot
from slavv_python.runtime.run_state import (
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
)
from slavv_python.runtime.run_tracking.status import target_stage_progress


def test_run_context_persists_snapshot_lifecycle(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        "slavv_python.runtime.run_tracking.context.RunContext._sample_process_memory_bytes",
        staticmethod(lambda: 4096),
    )
    context = build_run_context(
        run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="edges",
        provenance={"slavv_python": "unit-test"},
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
    assert snapshot.stages["preprocess"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].artifacts["checkpoint"].endswith("checkpoint_energy.pkl")
    assert snapshot.stages["energy"].artifacts["energy_field"] == "energy.npy"
    assert snapshot.optional_tasks["exports"].status == STATUS_COMPLETED
    assert snapshot.optional_tasks["exports"].artifacts["json"] == "network.json"
    assert target_stage_progress(snapshot) >= snapshot.overall_progress
    run_manifest = json.loads((run_dir / "99_Metadata" / "run_manifest.json").read_text("utf-8"))
    assert run_manifest["dataset_hash"] == "input-a"
    assert run_manifest["stage_metrics"]["energy"]["peak_memory_bytes"] == 4096
    assert run_manifest["stage_metrics"]["energy"]["elapsed_seconds"] >= 0.0


def test_from_existing_preserves_existing_target_stage(tmp_path):
    run_dir = tmp_path / "run"
    build_run_context(
        run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="edges",
    )

    reopened = RunContext.from_existing(run_dir)

    assert reopened.snapshot.target_stage == "edges"
    assert load_run_snapshot(run_dir).target_stage == "edges"


def test_stage_controller_rejects_unknown_stage(tmp_path):
    context = build_run_context(
        tmp_path / "run",
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )

    with pytest.raises(ValueError, match="stage must be one of"):
        context.stage("preprocess")
