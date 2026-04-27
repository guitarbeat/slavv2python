"""Tests for resume-guard behavior in structured runs."""

from __future__ import annotations

import pytest
from dev.tests.support.run_state_builders import build_run_context
from source.runtime import load_run_snapshot
from source.runtime._run_state.models import RunSnapshot
from source.runtime._run_state.resume_guard import (
    apply_resume_block,
    build_resume_block_message,
    fingerprint_mismatches,
    update_snapshot_fingerprints,
)
from source.runtime.run_state import STATUS_BLOCKED, STATUS_PENDING


def test_resume_guard_blocks_mismatched_input(tmp_path):
    run_dir = tmp_path / "blocked_run"
    context = build_run_context(
        run_dir,
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
    context = build_run_context(
        run_dir,
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


def test_fingerprint_mismatches_reports_changed_dimensions():
    snapshot = RunSnapshot(
        run_id="run-123",
        input_fingerprint="input-a",
        params_fingerprint="params-a",
    )

    mismatch = fingerprint_mismatches(
        snapshot,
        input_fingerprint="input-b",
        params_fingerprint="params-c",
    )

    assert mismatch == ["input", "parameters"]


def test_update_snapshot_fingerprints_overwrites_saved_values():
    snapshot = RunSnapshot(
        run_id="run-123",
        input_fingerprint="input-a",
        params_fingerprint="params-a",
    )

    update_snapshot_fingerprints(
        snapshot,
        input_fingerprint="input-b",
        params_fingerprint="params-c",
    )

    assert snapshot.input_fingerprint == "input-b"
    assert snapshot.params_fingerprint == "params-c"


def test_apply_resume_block_sets_blocked_status_and_error_message():
    snapshot = RunSnapshot(run_id="run-123")

    message = apply_resume_block(snapshot, ["input"])

    assert message == build_resume_block_message(["input"])
    assert snapshot.status == STATUS_BLOCKED
    assert "fingerprint changed" in snapshot.last_event
    assert snapshot.errors
