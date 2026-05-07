"""Tests for run-status parsing and rendering helpers."""

from __future__ import annotations

import calendar

from tests.support.run_state_builders import build_run_context

from slavv_python.runtime import RunContext
from slavv_python.runtime.run_tracking import build_status_lines


def test_parse_time_uses_utc_epoch():
    timestamp = "2026-03-27T12:00:00Z"

    parsed = RunContext._parse_time(timestamp)

    assert parsed == calendar.timegm((2026, 3, 27, 12, 0, 0))


def test_build_status_lines_include_optional_tasks(tmp_path):
    run_dir = tmp_path / "run"
    context = build_run_context(
        run_dir,
        input_fingerprint="input-a",
        params_fingerprint="params-a",
        target_stage="network",
    )
    context.update_optional_task(
        "export_network",
        status="running",
        detail="Writing JSON and CSV exports.",
        artifacts={
            "json": "C:\\temp\\network.json",
            "csv": "C:\\temp\\network.csv",
        },
    )

    lines = build_status_lines(context.snapshot)
    rendered = "\n".join(lines)

    assert "Optional tasks:" in rendered
    assert "export_network: running" in rendered
    assert "Writing JSON and CSV exports." in rendered
