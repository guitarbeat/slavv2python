"""Tests for snapshot store and event emission helpers."""

from __future__ import annotations

from dev.tests.support.run_state_builders import build_snapshot_dict, materialize_run_snapshot

from slavv.runtime._run_state.layout import resolve_run_layout
from slavv.runtime._run_state.models import RunSnapshot, StageSnapshot
from slavv.runtime._run_state.snapshot_store import emit_progress_event, load_or_create_snapshot


def test_load_or_create_snapshot_reuses_existing_snapshot(tmp_path):
    run_dir = tmp_path / "run"
    materialize_run_snapshot(
        run_dir,
        build_snapshot_dict(
            run_id="run-123",
            target_stage="edges",
            provenance={"source": "existing"},
        ),
    )
    layout = resolve_run_layout(run_dir=run_dir, checkpoint_dir=None, legacy=False)

    snapshot = load_or_create_snapshot(
        layout,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="vertices",
        provenance={"new": "value"},
    )

    assert snapshot.run_id == "run-123"
    assert snapshot.target_stage == "vertices"
    assert snapshot.provenance["source"] == "existing"
    assert snapshot.provenance["new"] == "value"


def test_emit_progress_event_uses_stage_progress_and_deep_copies_snapshot():
    snapshot = RunSnapshot(
        run_id="run-123",
        overall_progress=0.5,
        stages={"energy": StageSnapshot(name="energy", progress=0.25, resumed=True)},
    )
    seen = []

    emit_progress_event(
        snapshot,
        lambda event: seen.append(event),
        stage="energy",
        status="running",
        detail="Halfway",
    )

    assert len(seen) == 1
    event = seen[0]
    assert event.stage_progress == 0.25
    assert event.resumed is True
    assert event.snapshot is not snapshot
