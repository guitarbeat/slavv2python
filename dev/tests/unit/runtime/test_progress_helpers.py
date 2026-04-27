"""Tests for pure run-progress helper functions."""

from __future__ import annotations

import calendar

from source.runtime._run_state.models import RunSnapshot, StageSnapshot
from source.runtime._run_state.progress import (
    calculate_overall_progress,
    estimate_run_eta,
    estimate_stage_eta,
    parse_run_time,
    preprocess_complete,
)


def test_parse_run_time_uses_utc_epoch():
    timestamp = "2026-03-27T12:00:00Z"

    parsed = parse_run_time(timestamp)

    assert parsed == calendar.timegm((2026, 3, 27, 12, 0, 0))


def test_estimate_stage_eta_updates_elapsed_seconds():
    stage = StageSnapshot(
        name="energy",
        progress=0.5,
        started_at="2026-03-27T12:00:00Z",
    )

    eta = estimate_stage_eta(stage, now=calendar.timegm((2026, 3, 27, 12, 0, 10)))

    assert eta == 10.0
    assert stage.elapsed_seconds == 10.0


def test_estimate_run_eta_updates_snapshot_elapsed_seconds():
    snapshot = RunSnapshot(
        run_id="run-123",
        created_at="2026-03-27T12:00:00Z",
        overall_progress=0.25,
    )

    eta = estimate_run_eta(snapshot, now=calendar.timegm((2026, 3, 27, 12, 0, 20)))

    assert eta == 60.0
    assert snapshot.elapsed_seconds == 20.0


def test_preprocess_complete_checks_snapshot_artifacts_and_stage_status():
    stages = {"preprocess": StageSnapshot(name="preprocess")}
    snapshot = RunSnapshot(run_id="run-123", artifacts={"preprocess_done": "true"})

    assert preprocess_complete(stages, snapshot=snapshot) is True

    stages["preprocess"].status = "completed"
    assert preprocess_complete(stages, snapshot=None) is True


def test_calculate_overall_progress_uses_stage_weights_and_preprocess_flag():
    stages = {
        "energy": StageSnapshot(name="energy", progress=1.0),
        "vertices": StageSnapshot(name="vertices", progress=0.5),
        "edges": StageSnapshot(name="edges", progress=0.0),
        "network": StageSnapshot(name="network", progress=0.0),
    }

    progress = calculate_overall_progress(stages, preprocess_done=True)

    assert 0.0 < progress < 1.0
