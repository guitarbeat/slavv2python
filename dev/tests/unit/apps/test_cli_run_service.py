"""Tests for CLI run-command helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from source.apps.cli.run_service import (
    build_run_completion_lines,
    filter_export_formats,
    format_run_event_line,
    resolve_effective_run_dir,
)


def test_resolve_effective_run_dir_prefers_explicit_run_dir():
    assert resolve_effective_run_dir("out", "custom-run") == "custom-run"


def test_resolve_effective_run_dir_uses_output_when_not_legacy():
    assert resolve_effective_run_dir("out", None) == os.path.join("out", "_slavv_run")


def test_format_run_event_line_includes_detail():
    line = format_run_event_line(
        SimpleNamespace(
            stage="edges",
            stage_progress=0.25,
            overall_progress=0.5,
            detail="Tracing",
        )
    )

    assert "[edges]" in line
    assert "stage=25.0%" in line
    assert "overall=50.0%" in line
    assert "Tracing" in line


def test_filter_export_formats_all():
    assert filter_export_formats(["all"]) == ["json", "mat", "casx", "vmv"]


def test_filter_export_formats_individual():
    assert filter_export_formats(["json", "csv"]) == ["json"]


def test_build_run_completion_lines_includes_snapshot_lines():
    lines = build_run_completion_lines(
        SimpleNamespace(
            run_id="run-123",
            elapsed_seconds=12.3,
        ),
        {"json": "out/network.json"},
    )

    full_output = "".join(lines)
    assert "run-123" in full_output
    assert "12.3s" in full_output
    assert "JSON" in full_output
