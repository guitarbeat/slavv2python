"""Tests for legacy checkpoint snapshot bootstrapping."""

from __future__ import annotations

import pytest
from dev.tests.support.run_state_builders import build_run_context, materialize_checkpoint_surface

from slavv.runtime import load_run_snapshot
from slavv.runtime.run_state import (
    STATUS_COMPLETED,
    STATUS_PENDING,
    load_legacy_run_snapshot,
)


def test_legacy_checkpoints_bootstrap_snapshot(tmp_path):
    checkpoint_dir = tmp_path / "legacy_checkpoints"
    materialize_checkpoint_surface(
        checkpoint_dir,
        stages=("energy", "vertices"),
        structured=False,
    )

    context = build_run_context(
        checkpoint_dir,
        target_stage="network",
        legacy=True,
    )

    snapshot = context.snapshot
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["vertices"].status == STATUS_COMPLETED
    assert snapshot.stages["energy"].resumed is True
    assert snapshot.stages["edges"].status == STATUS_PENDING
    assert load_run_snapshot(checkpoint_dir) is not None


def test_load_legacy_run_snapshot_is_read_only(tmp_path):
    checkpoint_dir = tmp_path / "legacy_checkpoints"
    materialize_checkpoint_surface(
        checkpoint_dir,
        stages=("energy",),
        structured=False,
    )

    snapshot = load_legacy_run_snapshot(checkpoint_dir)
    repeated = load_legacy_run_snapshot(checkpoint_dir)

    assert snapshot is not None
    assert repeated is not None
    assert snapshot.stages["energy"].status == STATUS_COMPLETED
    assert snapshot.stages["preprocess"].status == STATUS_COMPLETED
    assert snapshot.overall_progress == pytest.approx(0.40)
    assert snapshot.run_id == repeated.run_id
    assert not (checkpoint_dir / "run_snapshot.json").exists()
    assert not (checkpoint_dir / "stage_state").exists()
