"""Integration tests for pipeline interactions with run-state persistence."""

from __future__ import annotations

import numpy as np
import pytest

from slavv.core import SLAVVProcessor
from slavv.runtime import load_run_snapshot
from slavv.runtime.run_state import STATUS_BLOCKED, STATUS_FAILED


def test_process_image_blocks_reuse_when_parameters_change(tmp_path):
    processor = SLAVVProcessor()
    image = np.zeros((4, 4, 4), dtype=np.float32)
    run_dir = tmp_path / "run"

    processor.process_image(
        image,
        {"radius_of_smallest_vessel_in_microns": 1.5},
        run_dir=str(run_dir),
        stop_after="energy",
    )
    first_snapshot = load_run_snapshot(run_dir)
    assert first_snapshot is not None
    original_params_fingerprint = first_snapshot.params_fingerprint

    with pytest.raises(RuntimeError, match="Resume blocked"):
        processor.process_image(
            image,
            {"radius_of_smallest_vessel_in_microns": 2.0},
            run_dir=str(run_dir),
            stop_after="energy",
        )

    blocked_snapshot = load_run_snapshot(run_dir)
    assert blocked_snapshot is not None
    assert blocked_snapshot.status == STATUS_BLOCKED
    assert blocked_snapshot.params_fingerprint == original_params_fingerprint


def test_process_image_rejects_invalid_stop_after():
    processor = SLAVVProcessor()
    image = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="stop_after must be one of"):
        processor.process_image(image, {}, stop_after="bogus")


def test_preprocess_failure_marks_run_failed(tmp_path, monkeypatch):
    processor = SLAVVProcessor()
    image = np.zeros((4, 4, 4), dtype=np.float32)
    run_dir = tmp_path / "run"

    def _boom(_image, _parameters):
        raise RuntimeError("preprocess exploded")

    monkeypatch.setattr("slavv.core.pipeline.utils.preprocess_image", _boom)

    with pytest.raises(RuntimeError, match="preprocess exploded"):
        processor.process_image(image, {}, run_dir=str(run_dir))

    snapshot = load_run_snapshot(run_dir)
    assert snapshot is not None
    assert snapshot.status == STATUS_FAILED
    assert snapshot.current_stage == "preprocess"
    assert snapshot.stages["preprocess"].status == STATUS_FAILED
