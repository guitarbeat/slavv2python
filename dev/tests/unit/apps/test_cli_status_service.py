"""Tests for CLI status-loading helpers."""

from __future__ import annotations

from source.apps.cli_status_service import build_status_output_lines, load_status_snapshot


def test_load_status_snapshot_prefers_primary_snapshot():
    snapshot = object()

    loaded = load_status_snapshot(
        "run-dir",
        snapshot_loader=lambda run_dir: snapshot if run_dir == "run-dir" else None,
    )

    assert loaded is snapshot


def test_load_status_snapshot_returns_none_when_missing():
    loaded = load_status_snapshot(
        "run-dir",
        snapshot_loader=lambda run_dir: None,
    )

    assert loaded is None


def test_build_status_output_lines_uses_status_line_builder():
    snapshot = object()

    lines = build_status_output_lines(
        snapshot,
        status_line_builder=lambda current_snapshot: [
            "line-a",
            f"same:{current_snapshot is snapshot}",
        ],
    )

    assert lines == ["line-a", "same:True"]
