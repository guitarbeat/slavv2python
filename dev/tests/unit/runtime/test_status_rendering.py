"""Tests for run-status parsing and rendering helpers."""

from __future__ import annotations

import calendar

from dev.tests.support.run_state_builders import build_run_context

from slavv.runtime import RunContext
from slavv.runtime.run_state import build_status_lines


def test_parse_time_uses_utc_epoch():
    timestamp = "2026-03-27T12:00:00Z"

    parsed = RunContext._parse_time(timestamp)

    assert parsed == calendar.timegm((2026, 3, 27, 12, 0, 0))


def test_build_status_lines_includes_matlab_resume_details(tmp_path):
    run_dir = tmp_path / "run"
    context = build_run_context(
        run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )
    context.update_optional_task(
        "matlab_status",
        status="failed",
        detail="Rerun will reuse batch_260401-140000 but restart energy from the stage boundary.",
        artifacts={
            "batch_folder": "C:\\temp\\batch_260401-140000",
            "resume_mode": "restart-current-stage",
            "last_completed_stage": "(none)",
            "next_stage": "energy",
            "rerun_prediction": (
                "Rerun will reuse batch_260401-140000 but restart energy from the "
                "stage boundary."
            ),
            "python_force_rerun_from": "edges",
            "failure_summary_file": "C:\\temp\\matlab_failure_summary.json",
        },
    )

    lines = build_status_lines(context.snapshot)
    rendered = "\n".join(lines)

    assert "MATLAB resume:" in rendered
    assert "Batch folder: C:\\temp\\batch_260401-140000" in rendered
    assert "Resume mode: restart-current-stage | last completed=(none) | next=energy" in rendered
    assert (
        "Prediction: Rerun will reuse batch_260401-140000 but restart energy from the stage boundary."
        in rendered
    )
    assert "Python rerun from: edges" in rendered
